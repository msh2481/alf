# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ValueNetwork and ValueRNNNetwork."""

import functools
import math
from typing import Callable

import torch
import torch.nn as nn

import alf
import alf.layers as layers
from alf.utils.perturb_utils import perturb_module_params_l2_sphere_per_layer
from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from .preprocessor_networks import PreprocessorNetwork
from alf.networks import Network
from alf.tensor_specs import TensorSpec
import alf.utils.math_ops as math_ops


@alf.configurable
class ValueNetworkBase(Network):
    """A base class for ``ValueNetwork`` and ``ValueRNNNetwork``.

    Can also be used to create customized value networks by providing
    different encoding network creators.
    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 output_tensor_spec: alf.NestedTensorSpec,
                 encoding_network_ctor: Callable,
                 last_kernel_initializer: Callable = None,
                 name="ValueNetworkBase",
                 **encoder_kwargs):
        """
        Args:
            input_tensor_spec: the tensor spec of the input.
            output_tensor_spec: spec for the value output.
            encoding_network_ctor: the creator of the encoding network that does
                the heavy lifting of the value network.
            last_kernel_initializer: initializer for the last layer. If None,
                defaults to uniform(-0.03, 0.03).
            name: name of the network
            encoder_kwargs: the extra keyword arguments to the encoding network
        """
        super().__init__(input_tensor_spec, name=name)

        if encoder_kwargs.get('kernel_initializer', None) is None:
            encoder_kwargs[
                'kernel_initializer'] = torch.nn.init.xavier_uniform_
        if last_kernel_initializer is None:
            last_kernel_initializer = functools.partial(torch.nn.init.uniform_,
                                                        a=-0.03,
                                                        b=0.03)

        self._encoding_net = encoding_network_ctor(
            input_tensor_spec=input_tensor_spec,
            last_layer_size=output_tensor_spec.numel,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer,
            **encoder_kwargs)
        self._output_spec = output_tensor_spec

    def forward(self, observation, state=()):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with ValueRNNNetwork

        Returns:
            value (torch.Tensor): a 1D tensor
            state: empty
        """
        value, state = self._encoding_net(observation, state)
        value = value.reshape(value.shape[0], *self._output_spec.shape)
        return value, state

    def make_parallel(self, n):
        """Create a ``ParallelValueNetwork`` using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        """
        return ParallelValueNetwork(self, n, "parallel_" + self._name)

    @property
    def state_spec(self):
        """Return the state spec of the value network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class ValueNetwork(ValueNetworkBase):
    """Output temporally uncorrelated values."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 input_preprocessors=None,
                 input_preprocessors_ctor=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 last_kernel_initializer=None,
                 use_fc_bn=False,
                 use_fc_ln=False,
                 name="ValueNetwork"):
        """Creates a value network that estimates the expected return.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            output_tensor_spec (TensorSpec): spec for the output
            input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            input_preprocessors_ctor (Callable): if ``input_preprocessors`` is None
                and ``input_preprocessors_ctor`` is provided, then ``input_preprocessors``
                will be constructed by calling ``input_preprocessors_ctor(input_tensor_spec)``.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default xavier_uniform
                initializer will be used.
            use_fc_bn (bool): whether use Batch Normalization for the internal
                FC layers (i.e. FC layers beside the last one).
            use_fc_ln (bool): whether use Layer Normalization for the internal
                fc layers (i.e. FC layers except the last one).
            name (str):
        """
        super().__init__(input_tensor_spec,
                         output_tensor_spec,
                         encoding_network_ctor=EncodingNetwork,
                         last_kernel_initializer=last_kernel_initializer,
                         name=name,
                         input_preprocessors=input_preprocessors,
                         input_preprocessors_ctor=input_preprocessors_ctor,
                         preprocessing_combiner=preprocessing_combiner,
                         conv_layer_params=conv_layer_params,
                         fc_layer_params=fc_layer_params,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         use_fc_bn=use_fc_bn,
                         use_fc_ln=use_fc_ln)


class ParallelValueNetwork(Network):
    """Perform ``n`` value computations in parallel."""

    def __init__(self,
                 value_network: ValueNetwork,
                 n: int,
                 name="ParallelValueNetwork"):
        """
        It creates a parallelized version of ``value_network``.
        Args:
            value_network (ValueNetwork): non-parallelized value network
            n (int): make ``n`` replicas from ``value_network`` with different
                initialization.
            name (str):
        """

        super().__init__(input_tensor_spec=value_network.input_tensor_spec,
                         name=name)
        self._encoding_net = value_network._encoding_net.make_parallel(n, True)
        self._output_spec = TensorSpec((n, ) + value_network.output_spec.shape)

    def forward(self, observation, state=()):
        """Computes values given a batch of observations.
        Args:
            inputs (tuple):  A tuple of Tensors consistent with `input_tensor_spec``.
            state (tuple): Empty for API consistent with ``ValueRNNNetwork``.
        """

        value, state = self._encoding_net(observation, state)
        value = value.reshape(value.shape[0], *self._output_spec.shape)
        return value, state

    @property
    def state_spec(self):
        """Return the state spec of the value network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class ValueRNNNetwork(ValueNetworkBase):
    """Outputs temporally correlated values."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 value_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="ValueRNNNetwork"):
        """Creates an instance of `ValueRNNNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            output_tensor_spec (TensorSpec): spec for the output
            input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as math_ops.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            value_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default xavier_uniform
                initializer will be used.
            name (str):
        """
        super().__init__(input_tensor_spec=input_tensor_spec,
                         output_tensor_spec=output_tensor_spec,
                         encoding_network_ctor=LSTMEncodingNetwork,
                         name=name,
                         input_preprocessors=input_preprocessors,
                         preprocessing_combiner=preprocessing_combiner,
                         conv_layer_params=conv_layer_params,
                         pre_fc_layer_params=fc_layer_params,
                         hidden_size=lstm_hidden_size,
                         post_fc_layer_params=value_fc_layer_params,
                         activation=activation,
                         kernel_initializer=kernel_initializer)


@alf.configurable
class RandomizedPriorValueNetwork(Network):
    """A ValueNetwork augmented with a randomized prior function.

    Mirrors ``RandomizedPriorCriticNetwork`` but for V(s) networks.
    Two instances of ``network_ctor`` are created: one trainable and one frozen
    prior. Their outputs are summed. The last FC layer of each is re-initialized
    post-construction to achieve the desired output scale, since ``ValueNetwork``
    does not expose ``last_kernel_initializer`` as a constructor argument.
    """

    def __init__(self,
                 network_ctor: Callable = ValueNetwork,
                 input_tensor_spec=None,
                 prior_scale: float = 1.0,
                 trainable_init_std: float = 1e-3,
                 name="RandomizedPriorValueNetwork",
                 **network_kwargs):
        """
        Args:
            network_ctor: Constructor for the base value network (e.g., ValueNetwork).
            input_tensor_spec: TensorSpec of the observation input.
            prior_scale: Desired output std of the frozen prior network.
            trainable_init_std: Desired output std of the trainable network at init.
            name: Name of the network.
            **network_kwargs: Extra kwargs forwarded to ``network_ctor``.
        """
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        # Compute last-layer input dim to scale weights so output has desired std.
        # output_std = weight_std * sqrt(input_dim), so weight_std = output_std / sqrt(input_dim)
        temp_net = network_ctor(input_tensor_spec=input_tensor_spec,
                                **network_kwargs)
        last_fc_dim = self._get_last_layer_input_dim(temp_net)
        del temp_net

        trainable_init = functools.partial(torch.nn.init.normal_,
                                           std=3 * trainable_init_std /
                                           math.sqrt(last_fc_dim))
        prior_init = functools.partial(torch.nn.init.normal_,
                                       std=3 * prior_scale /
                                       math.sqrt(last_fc_dim))

        self._trainable_net = network_ctor(
            input_tensor_spec=input_tensor_spec,
            last_kernel_initializer=trainable_init,
            **network_kwargs)
        self._prior_net = network_ctor(input_tensor_spec=input_tensor_spec,
                                       last_kernel_initializer=prior_init,
                                       **network_kwargs)

        for param in self._prior_net.parameters():
            param.requires_grad = False

    def _get_last_layer_input_dim(self, net):
        last_fc = None
        for module in net.modules():
            if isinstance(module, layers.FC):
                last_fc = module
        return last_fc.weight.shape[1] if last_fc else 1

    def forward(self, observation, state=()):
        v, state = self._trainable_net(observation, state)
        prior_v, _ = self._prior_net(observation, state)
        return v + prior_v, state

    def perturb_prior(self, alpha: float):
        """Perturb frozen prior parameters with a per-layer L2-sphere walk."""
        perturb_module_params_l2_sphere_per_layer(self._prior_net, alpha)

    @property
    def state_spec(self):
        return self._trainable_net.state_spec

    def make_parallel(self, n):
        parallel_trainable = self._trainable_net.make_parallel(n)
        parallel_prior = alf.networks.NaiveParallelNetwork(self._prior_net, n)
        for p in parallel_prior.parameters():
            p.requires_grad = False
        return _ParallelRandomizedPriorValueNetwork(parallel_trainable,
                                                    parallel_prior,
                                                    self.input_tensor_spec)


class _ParallelRandomizedPriorValueNetwork(Network):
    """Parallel version of RandomizedPriorValueNetwork."""

    def __init__(self,
                 parallel_trainable,
                 parallel_prior,
                 input_tensor_spec,
                 name="ParallelRandomizedPriorValueNetwork"):
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)
        self._trainable_net = parallel_trainable
        self._prior_net = parallel_prior
        self._output_spec = parallel_trainable.output_spec
        for p in self._prior_net.parameters():
            p.requires_grad = False

    def forward(self, observation, state=()):
        v, state = self._trainable_net(observation, state)
        prior_v, _ = self._prior_net(observation, state)
        return v + prior_v, state

    def perturb_prior(self, alpha: float):
        nets = getattr(self._prior_net, "_networks", None)
        if nets is not None:
            for net in nets:
                perturb_module_params_l2_sphere_per_layer(net, alpha)
        else:
            perturb_module_params_l2_sphere_per_layer(self._prior_net, alpha)

    @property
    def state_spec(self):
        return self._trainable_net.state_spec
