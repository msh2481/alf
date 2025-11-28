# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""QNetworks"""

import functools
import math
from typing import Callable

import torch
import torch.nn as nn

import alf
import alf.nest as nest
import alf.layers as layers
from alf.networks import EncodingNetwork, LSTMEncodingNetwork, ParallelEncodingNetwork
from alf.networks import Network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
import alf.utils.math_ops as math_ops


@alf.configurable
class QNetworkBase(Network):
    """A base class for ``QNetwork`` and ``QRNNNetwork``.

    Can also be used to create customized value networks by providing
    different encoding network creators.
    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 action_spec: BoundedTensorSpec,
                 encoding_network_ctor: Callable,
                 use_naive_parallel_network: bool = False,
                 last_kernel_initializer=None,
                 name: str = "QNetworkBase",
                 bias_init_value: float = -0.2,
                 **encoder_kwargs):
        """
        Args:
            input_tensor_spec: the tensor spec of the input
            action_spec : the tensor spec of the action
            encoding_network_ctor: the creator of the encoding network that does
                the heavy lifting of the q network.
            use_naive_parallel_network: if True, will use
                ``NaiveParallelNetwork`` when ``make_parallel`` is called. This
                might be useful in cases when the ``NaiveParallelNetwork``
                has an advantange in terms of speed over ``ParallelNetwork``.
                You have to test to see which way is faster for your particular
                situation.
            last_kernel_initializer: initializer for the final layer weights.
                If None, uses uniform initialization with range [-0.003, 0.003].
            name: name of the network
            encoder_kwargs: the extra keyword arguments to the encoding network
        """
        super().__init__(input_tensor_spec, name=name)

        assert len(nest.flatten(action_spec)) == 1, (
            "Currently only support a single discrete action! Use "
            "CriticNetwork instead for multiple actions.")

        num_actions = action_spec.maximum - action_spec.minimum + 1

        self._use_naive_parallel_network = use_naive_parallel_network
        self._output_spec = TensorSpec((num_actions, ))

        self._encoding_net = encoding_network_ctor(
            input_tensor_spec=input_tensor_spec, **encoder_kwargs)

        if last_kernel_initializer is None:
            last_kernel_initializer = functools.partial(torch.nn.init.uniform_,
                                                        a=-0.003,
                                                        b=0.003)

        self._final_layer = layers.FC(
            self._encoding_net.output_spec.shape[0],
            num_actions,
            activation=math_ops.identity,
            kernel_initializer=last_kernel_initializer,
            bias_init_value=bias_init_value)

    def forward(self, observation, state=()):
        """Computes action values given an observation.

        Args:
            observation (nest): consistent with ``input_tensor_spec``
            state: empty for API consistent with ``QRNNNetwork``

        Returns:
            tuple:
            - action_value (torch.Tensor): a tensor of the size
              ``[batch_size, num_actions]``
            - state: empty
        """
        encoded_obs, state = self._encoding_net(observation, state)
        action_value = self._final_layer(encoded_obs)
        return action_value, state

    def make_parallel(self, n):
        """Create a ``ParallelQNetwork`` using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        If ``use_naive_parallel_network`` is True, use ``NaiveParallelNetwork``
        to create the parallel network.
        """
        if self._use_naive_parallel_network:
            return alf.networks.NaiveParallelNetwork(self, n)
        else:
            return ParallelQNetwork(self, n, "parallel_" + self._name)

    @property
    def state_spec(self):
        """Return the state spec of the q network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class QNetwork(QNetworkBase):
    """Create an instance of QNetwork."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_naive_parallel_network=False,
                 last_kernel_initializer=None,
                 bias_init_value: float = -0.2,
                 name="QNetwork"):
        """Creates an instance of ``QNetwork`` for estimating action-value of
        discrete actions. The action-value is defined as the expected return
        starting from the given input observation and taking the given action.
        It takes observation as input and outputs an action-value tensor with
        the shape of ``[batch_size, num_of_actions]``.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
            input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as ``math_ops.identity``.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see ``alf.nest.utils.NestConcat``. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default ``variance_scaling_initializer``
                will be used.
            use_naive_parallel_network (bool): if True, will use
                ``NaiveParallelNetwork`` when ``make_parallel`` is called. This
                might be useful in cases when the ``NaiveParallelNetwork``
                has an advantange in terms of speed over ``ParallelNetwork``.
                You have to test to see which way is faster for your particular
                situation.
            last_kernel_initializer: initializer for the final layer weights.
                If None, uses uniform initialization with range [-0.003, 0.003].
        """
        super(QNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            encoding_network_ctor=EncodingNetwork,
            use_naive_parallel_network=use_naive_parallel_network,
            last_kernel_initializer=last_kernel_initializer,
            name=name,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_init_value=bias_init_value)


class ParallelQNetwork(Network):
    """Perform ``n`` Q-value computations in parallel."""

    def __init__(self, q_network: QNetwork, n: int, name="ParallelQNetwork"):
        """
        Args:
            q_network (QNetwork): non-parallelized q network
            n (int): make ``n`` replicas from ``q_network`` with different
                parameter initializations.
            name (str):
        """
        super().__init__(input_tensor_spec=q_network.input_tensor_spec,
                         name=name)
        self._encoding_net = q_network._encoding_net.make_parallel(n, True)
        self._final_layer = q_network._final_layer.make_parallel(n)
        self._output_spec = TensorSpec((n, ) +
                                       tuple(q_network.output_spec.shape))

    def forward(self, inputs, state=()):
        """Compute action values given an observation.

        Args:
            inputs (nest): consistent with ``input_tensor_spec``.
            state: empty for API consistent with ``QRNNNetwork``.

        Returns:
            tuple:
            - action_value (Tensor): a tensor of shape :math:`[B,n,k]`, where
              :math:`B` is the batch size, :math:`n` is the num of replicas, and
              :math:`k` is the number of actions.
            - state: empty
        """
        encoded_obs, state = self._encoding_net(inputs, state)
        action_value = self._final_layer(encoded_obs)
        return action_value, state

    @property
    def state_spec(self):
        """Return the state spec of the q network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class QRNNNetwork(QNetworkBase):
    """Create a RNN-based that outputs temporally correlated q-values."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 value_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_naive_parallel_network=False,
                 name="QRNNNetwork"):
        """Creates an instance of `QRNNNetwork` for estimating action-value of
        discrete actions. The action-value is defined as the expected return
        starting from the given inputs (observation and state) and taking the
        given action. It takes observation and state as input and outputs an
        action-value tensor with the shape of [batch_size, num_of_actions].
        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
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
            value_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default
                variance_scaling_initializer will be used.
            use_naive_parallel_network (bool): if True, will use
                ``NaiveParallelNetwork`` when ``make_parallel`` is called. This
                might be useful in cases when the ``NaiveParallelNetwork``
                has an advantange in terms of speed over ``ParallelNetwork``.
                You have to test to see which way is faster for your particular
                situation.
        """
        super(QRNNNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            encoding_network_ctor=LSTMEncodingNetwork,
            use_naive_parallel_network=use_naive_parallel_network,
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
class RandomizedPriorQNetwork(Network):
    """A Q-Network augmented with a randomized prior function.

    This network creates two instances of the same Q-network architecture:
    one trainable network and one frozen prior network. The outputs are summed
    to give the final Q-values. This implements the randomized prior functions
    technique for improved exploration.

    Can wrap any Q-network type (QNetwork, QRNNNetwork, DebugLinearQNetwork, etc.)
    """

    def __init__(self,
                 network_ctor: Callable,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 prior_scale: float = 1.0,
                 trainable_init_std: float = 1e-3,
                 name="RandomizedPriorQNetwork",
                 bias_init_value: float = 0.0,
                 **network_kwargs):
        """Creates a Q-Network with randomized prior.

        Args:
            network_ctor: Constructor for the base Q-network (e.g., QNetwork,
                DebugLinearQNetwork)
            input_tensor_spec: the tensor spec of the input
            action_spec: the tensor spec of the action
            prior_scale: Target standard deviation for the prior network's output.
                The weight initialization is scaled so the output has this std.
            trainable_init_std: Target standard deviation for the trainable
                network's output. Typically small (e.g., 1e-3) so the network
                starts near zero.
            name: name of the network
            **network_kwargs: additional arguments passed to network_ctor
        """
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        # Create temp network to get last layer input dim
        temp_net = network_ctor(input_tensor_spec=input_tensor_spec,
                                action_spec=action_spec,
                                **network_kwargs)
        last_layer_input_dim = self._get_last_layer_input_dim(temp_net)
        del temp_net

        # Scale weights so OUTPUT has desired std
        trainable_weight_std = trainable_init_std / math.sqrt(
            last_layer_input_dim)
        prior_weight_std = prior_scale / math.sqrt(last_layer_input_dim)

        # Trainable network with small initialization
        trainable_init = functools.partial(torch.nn.init.normal_,
                                           std=trainable_weight_std)
        self._trainable_net = network_ctor(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            last_kernel_initializer=trainable_init,
            bias_init_value=0.0,
            **network_kwargs)

        # Prior network with larger random initialization
        prior_init = functools.partial(torch.nn.init.normal_,
                                       std=prior_weight_std)
        self._prior_net = network_ctor(input_tensor_spec=input_tensor_spec,
                                       action_spec=action_spec,
                                       last_kernel_initializer=prior_init,
                                       bias_init_value=bias_init_value,
                                       **network_kwargs)

        # Freeze the prior network
        for param in self._prior_net.parameters():
            param.requires_grad = False

        self._output_spec = self._trainable_net.output_spec

    def _get_last_layer_input_dim(self, net):
        """Find the input dimension of the last FC layer."""
        last_fc = None
        for module in net.modules():
            if isinstance(module, layers.FC):
                last_fc = module
        return last_fc.weight.shape[1] if last_fc else 1

    def forward(self, observation, state=()):
        """Computes action values by summing trainable network and prior.

        Args:
            observation (nest): consistent with input_tensor_spec
            state: network state (for RNN-based networks)

        Returns:
            tuple:
            - action_value (torch.Tensor): sum of trainable and prior Q-values
            - state: updated state
        """
        q_vals, state = self._trainable_net(observation, state)
        with torch.no_grad():
            prior_vals, _ = self._prior_net(observation, state)
        return q_vals + prior_vals, state

    def make_parallel(self, n):
        """Create parallel version using NaiveParallelNetwork.

        Each replica will have its own independent trainable and prior networks.
        """
        return alf.networks.NaiveParallelNetwork(self, n)

    @property
    def state_spec(self):
        """Return the state spec (delegates to trainable network)."""
        return self._trainable_net.state_spec

    def _log_parameters(self):
        input_dim = self.input_tensor_spec.shape[0]
        batch_size = input_dim + 1
        inputs = torch.zeros(batch_size, input_dim)
        for i in range(input_dim):
            inputs[i + 1, i] = 1.0

        with torch.no_grad():
            outputs, _ = self.forward(inputs)

        bias = outputs[0]
        coeffs_plus_bias = outputs[1:]

        lines = []
        totals = []
        for action_idx in range(bias.shape[0]):
            bias_val = bias[action_idx].item()
            weights = coeffs_plus_bias[:, action_idx]
            weight_str = ', '.join([f"{val:.2f}" for val in weights])
            lines.append(
                f"Action {action_idx}: [{weight_str}] (bias = {bias_val:.2f})")
            totals.append(weights)

        if len(totals) == 2:
            delta = totals[1] - totals[0]
            delta_str = ', '.join([f"{val:.2f}" for val in delta])
            lines.append(f"Delta: [{delta_str}]")

        return '\n'.join(lines)


@alf.configurable
class OptimisticQNetwork(QNetwork):
    """A Q-Network with configurable optimistic initialization.

    This network allows control over the initialization of the final layer
    through mean and std parameters, useful for implementing optimistic
    initialization strategies.
    """

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 init_mean: float = 0.0,
                 init_std: float = 1.0,
                 name="OptimisticQNetwork",
                 use_naive_parallel_network=False,
                 **kwargs):
        """Creates a Q-Network with optimistic initialization.

        Args:
            input_tensor_spec: the tensor spec of the input
            action_spec: the tensor spec of the action
            init_mean: mean for initializing final layer weights
            init_std: standard deviation for initializing final layer weights
            name: name of the network
            **kwargs: additional arguments passed to QNetwork
        """
        super(OptimisticQNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            name=name,
            use_naive_parallel_network=use_naive_parallel_network,
            **kwargs)

        num_actions = (action_spec.maximum - action_spec.minimum + 1).item()
        custom_kernel_initializer = functools.partial(torch.nn.init.normal_,
                                                      mean=init_mean,
                                                      std=init_std)
        self._final_layer = layers.FC(
            self._encoding_net.output_spec.shape[0],
            num_actions,
            activation=math_ops.identity,
            kernel_initializer=custom_kernel_initializer,
            bias_init_value=0.0)

    def _log_parameters(self):
        input_dim = self.input_tensor_spec.shape[0]
        batch_size = input_dim + 1
        inputs = torch.zeros(batch_size, input_dim)
        for i in range(input_dim):
            inputs[i + 1, i] = 1.0
        with torch.no_grad():
            outputs, _ = self.forward(inputs)
        bias = outputs[0]
        coeffs_plus_bias = outputs[1:]

        lines = []
        totals = []
        for action_idx in range(bias.shape[0]):
            bias_val = bias[action_idx].item()
            weights = coeffs_plus_bias[:, action_idx]
            weight_str = ', '.join([f"{val:.2f}" for val in weights])
            lines.append(
                f"Action {action_idx}: [{weight_str}] (bias = {bias_val:.2f})")
            totals.append(weights)

        if len(totals) == 2:
            delta = totals[1] - totals[0]
            delta_str = ', '.join([f"{val:.2f}" for val in delta])
            lines.append(f"Delta: [{delta_str}]")

        return '\n'.join(lines)


@alf.configurable
class DebugLinearQNetwork(QNetworkBase):
    """A purely linear QNetwork with periodic parameter logging.

    This network uses an IdentityEncodingNetwork, making it equivalent to
    a single linear layer from observations to Q-values. Useful for debugging
    on environments where linear function approximation is sufficient.

    Periodically logs parameter statistics to help understand network behavior.
    """

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 name="DebugLinearQNetwork"):
        """Creates a linear QNetwork with parameter logging.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
            log_frequency (float): probability of logging on each forward pass
                (default 0.01 = 1% of forward passes)
            name (str): name of the network
        """
        # Import here to avoid circular dependency
        from alf.networks.encoding_networks import IdentityEncodingNetwork

        super(DebugLinearQNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            encoding_network_ctor=IdentityEncodingNetwork,
            use_naive_parallel_network=False,
            name=name,
        )
        self._final_layer = layers.FC(
            input_size=input_tensor_spec.shape[0],
            output_size=action_spec.maximum - action_spec.minimum + 1,
            activation=math_ops.identity,
            kernel_initializer=torch.nn.init.normal_,
            # bias_init_value=0.0,
            use_bias=False)
        self._forward_count = 0

    def forward(self, observation, state=()):
        self._forward_count += 1
        return super().forward(observation, state)

    def make_parallel(self, n):
        return alf.networks.NaiveParallelNetwork(self, n)

    def _log_parameters(self):
        totals = []
        for i in range(self._final_layer.weight.shape[0]):
            bias_val = self._final_layer.bias[i].data.item()
            total = bias_val + self._final_layer.weight[i].data.flatten()
            weight_str = ', '.join([f"{val:.2f}" for val in total])
            print(f"Action {i}: [{weight_str}] (bias = {bias_val:.2f})")
            totals.append(total)
        if len(totals) == 2:
            delta = totals[1] - totals[0]
            delta_str = ', '.join([f"{val:.2f}" for val in delta])
            print(f"Delta: [{delta_str}]")
