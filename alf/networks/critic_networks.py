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
"""CriticNetworks"""

import functools
import math
from typing import Callable

import torch

import alf
import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec

from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork, ParallelEncodingNetwork, RBFEncodingNetwork
from .network import Network
from .preprocessors import CosineEmbeddingPreprocessor
import alf.layers as layers


def _check_action_specs_for_critic_networks(action_spec,
                                            action_input_processors,
                                            action_preprocessing_combiner):

    if len(nest.flatten(action_spec)) > 1:
        assert action_preprocessing_combiner is not None, (
            "An action combiner is needed when there are multiple action specs:"
            " {}".format(action_spec))

    def _check_individual(spec, proc):
        if spec.is_discrete:
            assert proc is not None, (
                'CriticNetwork only supports continuous actions. One of given '
                + 'action specs {} is discrete. Use QNetwork instead. '.format(
                    spec) +
                'Alternatively, specify `action_input_processors` to transform '
                + 'discrete actions to continuous action embeddings first.')

    if action_input_processors is None:
        action_input_processors = nest.map_structure(lambda _: None,
                                                     action_spec)

    nest.map_structure(_check_individual, action_spec, action_input_processors)


@alf.configurable
class CriticNetwork(EncodingNetwork):
    """Creates an instance of ``CriticNetwork`` for estimating action-value of
    continuous or discrete actions. The action-value is defined as the expected
    return starting from the given input observation and taking the given action.
    This module takes observation as input and action as input and outputs an
    action-value tensor with the shape of ``[batch_size]``.

    The network take a tuple of (observation, action) as input to computes the
    action-value given an observation.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 observation_input_processors=None,
                 observation_input_processors_ctor=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_processors=None,
                 action_input_processors_ctor=None,
                 action_preprocessing_combiner=None,
                 action_fc_layer_params=None,
                 observation_action_combiner=None,
                 joint_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 use_fc_ln=False,
                 last_use_fc_bn=False,
                 last_use_fc_ln=False,
                 last_layer_activation=math_ops.identity,
                 last_kernel_initializer=None,
                 use_naive_parallel_network=False,
                 name="CriticNetwork"):
        """

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            output_tensor_spec (TensorSpec): spec for the output
            observation_input_processors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding observation input.
            observation_input_processors_ctor (Callable): if ``observation_input_processors``
                is None and ``observation_input_processors_ctor`` is provided, then
                ``observation_input_processors`` will be constructed by calling
                ``observation_input_processors_ctor(observation_spec)``.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_processors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding action input.
            action_input_processors_ctor (Callable): if ``action_input_processors``
                is None and ``action_input_processors_ctor`` is provided, then
                ``action_input_processors`` will be constructed by calling
                ``action_input_processors_ctor(action_spec)``.
            action_preprocessing_combiner (NestCombiner): preprocessing called
                to combine complex action inputs.
            action_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for actions.
            observation_action_combiner (NestCombiner): combiner class for fusing
                the observation and action. If None, ``NestConcat`` will be used.
            joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            use_fc_bn (bool): whether use Batch Normalization for the internal
                FC layers (i.e. FC layers beside the last one).
            use_fc_ln (bool): whether use Layer Normalization for the internal
                FC layers (i.e. FC layers beside the last one).
            last_kernel_initializer (Callable): initializer for the last layer.
                If None, defaults to uniform initialization in [-0.003, 0.003].
            use_naive_parallel_network (bool): if True, will use
                ``NaiveParallelNetwork`` when ``make_parallel`` is called. This
                might be useful in cases when the ``NaiveParallelNetwork``
                has an advantange in terms of speed over ``ParallelNetwork``.
                You have to test to see which way is faster for your particular
                situation.
            name (str):
        """
        if kernel_initializer is None:
            kernel_initializer = functools.partial(variance_scaling_init,
                                                   gain=math.sqrt(1.0 / 3),
                                                   mode='fan_in',
                                                   distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessors=observation_input_processors,
            input_preprocessors_ctor=observation_input_processors_ctor,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            use_fc_ln=use_fc_ln,
            name=name + ".obs_encoder")

        _check_action_specs_for_critic_networks(action_spec,
                                                action_input_processors,
                                                action_preprocessing_combiner)
        action_encoder = EncodingNetwork(
            action_spec,
            input_preprocessors=action_input_processors,
            input_preprocessors_ctor=action_input_processors_ctor,
            preprocessing_combiner=action_preprocessing_combiner,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            use_fc_ln=use_fc_ln,
            name=name + ".action_encoder")

        if last_kernel_initializer is None:
            last_kernel_initializer = functools.partial(torch.nn.init.uniform_,
                                                        a=-0.003,
                                                        b=0.003)

        if observation_action_combiner is None:
            observation_action_combiner = alf.layers.NestConcat(dim=-1)

        super().__init__(input_tensor_spec=input_tensor_spec,
                         output_tensor_spec=output_tensor_spec,
                         input_preprocessors=(obs_encoder, action_encoder),
                         preprocessing_combiner=observation_action_combiner,
                         fc_layer_params=joint_fc_layer_params,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         use_fc_bn=use_fc_bn,
                         use_fc_ln=use_fc_ln,
                         last_layer_size=output_tensor_spec.numel,
                         last_activation=last_layer_activation,
                         last_kernel_initializer=last_kernel_initializer,
                         last_use_fc_bn=last_use_fc_bn,
                         last_use_fc_ln=last_use_fc_ln,
                         name=name)
        self._use_naive_parallel_network = use_naive_parallel_network

    def make_parallel(self, n):
        """Create a parallel critic network using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        If ``use_naive_parallel_network`` is True, use ``NaiveParallelNetwork``
        to create the parallel network.
        """
        if self._use_naive_parallel_network:
            return alf.networks.NaiveParallelNetwork(self, n)
        else:
            return super().make_parallel(n, True)


@alf.configurable
class CriticRNNNetwork(LSTMEncodingNetwork):
    """Creates an instance of ``CriticRNNNetwork`` for estimating action-value
    of continuous or discrete actions. The action-value is defined as the
    expected return starting from the given inputs (observation and state) and
    taking the given action. It takes observation and state as input and outputs
    an action-value tensor with the shape of [batch_size].
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 observation_input_processors=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_processors=None,
                 action_preprocessing_combiner=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 lstm_hidden_size=100,
                 critic_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="CriticRNNNetwork"):
        """

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            ourput_tensor_spec (TensorSpec): spec for the output
            observation_input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding observation input.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_processors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding action input.a
            action_preprocessing_combiner (NestCombiner): preprocessing called
                to combine complex action inputs.
            action_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for actions.
            joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            critic_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a ``variance_scaling_initializer``
                with uniform distribution will be used.
            name (str):
        """
        if kernel_initializer is None:
            kernel_initializer = functools.partial(variance_scaling_init,
                                                   gain=math.sqrt(1.0 / 3),
                                                   mode='fan_in',
                                                   distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessors=observation_input_processors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        _check_action_specs_for_critic_networks(action_spec,
                                                action_input_processors,
                                                action_preprocessing_combiner)
        action_encoder = EncodingNetwork(
            action_spec,
            input_preprocessors=action_input_processors,
            preprocessing_combiner=action_preprocessing_combiner,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_kernel_initializer = functools.partial(torch.nn.init.uniform_,
                                                    a=-0.003,
                                                    b=0.003)

        super().__init__(input_tensor_spec=input_tensor_spec,
                         output_tensor_spec=output_tensor_spec,
                         input_preprocessors=(obs_encoder, action_encoder),
                         preprocessing_combiner=alf.layers.NestConcat(dim=-1),
                         pre_fc_layer_params=joint_fc_layer_params,
                         hidden_size=lstm_hidden_size,
                         post_fc_layer_params=critic_fc_layer_params,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         last_layer_size=output_tensor_spec.numel,
                         last_activation=math_ops.identity,
                         last_kernel_initializer=last_kernel_initializer)

    def make_parallel(self, n):
        """Create a parallel critic RNN network using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        If ``use_naive_parallel_network`` is True, use ``NaiveParallelNetwork``
        to create the parallel network.
        """
        return super().make_parallel(n, True)


@alf.configurable
class CriticQuantileNetwork(EncodingNetwork):
    """Creates an instance of ``CriticQuantileNetwork`` for estimating the quantiles 
    of a (state, action) input for continuous or discrete actions. Used by the 
    DSacAlgorithm.
    """

    def __init__(self,
                 input_tensor_spec,
                 tau_spec,
                 output_tensor_spec=TensorSpec(()),
                 observation_input_processors=None,
                 observation_input_processors_ctor=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_processors=None,
                 action_input_processors_ctor=None,
                 action_preprocessing_combiner=None,
                 action_fc_layer_params=None,
                 observation_action_combiner=None,
                 obs_act_joint_fc_layer_params=None,
                 obs_act_activation=torch.relu_,
                 tau_embedding_dim=64,
                 tau_input_processors=None,
                 tau_fc_layer_params=None,
                 tau_activation=torch.sigmoid_,
                 obs_act_tau_joint_fc_layer_params=None,
                 use_fc_bn=False,
                 use_fc_ln=True,
                 kernel_initializer=None,
                 last_kernel_initializer=None,
                 use_naive_parallel_network=False,
                 name="CriticQuantileNetwork"):
        """

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            tau_spec (TensorSpec): spec for the tau input.
            output_tensor_spec (TensorSpec): spec for the output
            observation_input_processors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding observation input.
            observation_input_processors_ctor (Callable): if ``observation_input_processors``
                is None and ``observation_input_processors_ctor`` is provided, then
                ``observation_input_processors`` will be constructed by calling
                ``observation_input_processors_ctor(observation_spec)``.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_processors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding action input.
            action_input_processors_ctor (Callable): if ``action_input_processors``
                is None and ``action_input_processors_ctor`` is provided, then
                ``action_input_processors`` will be constructed by calling
                ``action_input_processors_ctor(action_spec)``.
            action_preprocessing_combiner (NestCombiner): preprocessing called
                to combine complex action inputs.
            action_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for actions.
            observation_action_combiner (NestCombiner): combiner class for fusing
                the observation and action. If None, ``NestConcat`` will be used.
            obs_act_joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            obs_act_activation (nn.functional): activation used for hidden layers after
                merging observations and actions.
            tau_embedding_dim (int): dimension of the tau embeddings.
            tau_input_processors (Network|nn.Module|None): input preprocessors applied
                to the input tau.
            tau_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes for the tau embedding.
            tau_activation (nn.functional): activation used for hidden layers of
                tau embedding.
            obs_act_tau_joint_fc_layer_params (tuple[int]): a tuple of integers
                representing hidden layers after merging the observation, action, and
                tau embedding.
            use_fc_bn (bool): whether use Batch Normalization for the internal
                FC layers (i.e. FC layers beside the last one).
            use_fc_ln (bool): whether use Layer Normalization for the internal
                FC layers (i.e. FC layers beside the last one).
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            last_kernel_initializer (Callable): initializer for all the last layer
                If none is provided a uniform initializer will be used.
            use_naive_parallel_network (bool): if True, will use
                ``NaiveParallelNetwork`` when ``make_parallel`` is called. This
                might be useful in cases when the ``NaiveParallelNetwork``
                has an advantange in terms of speed over ``ParallelNetwork``.
                You have to test to see which way is faster for your particular
                situation.
            name (str):
        """

        if kernel_initializer is None:
            kernel_initializer = functools.partial(variance_scaling_init,
                                                   gain=math.sqrt(1.0 / 3),
                                                   mode='fan_in',
                                                   distribution='uniform')

        if last_kernel_initializer is None:
            last_kernel_initializer = functools.partial(torch.nn.init.uniform_,
                                                        a=-0.003,
                                                        b=0.003)

        obs_act_encoder = CriticNetwork(
            input_tensor_spec,
            output_tensor_spec=TensorSpec(
                (1, obs_act_tau_joint_fc_layer_params[0])),
            observation_input_processors=observation_input_processors,
            observation_input_processors_ctor=observation_input_processors_ctor,
            observation_preprocessing_combiner=
            observation_preprocessing_combiner,
            observation_conv_layer_params=observation_conv_layer_params,
            observation_fc_layer_params=observation_fc_layer_params,
            action_input_processors=action_input_processors,
            action_input_processors_ctor=action_input_processors_ctor,
            action_preprocessing_combiner=action_preprocessing_combiner,
            action_fc_layer_params=action_fc_layer_params,
            observation_action_combiner=observation_action_combiner,
            joint_fc_layer_params=obs_act_joint_fc_layer_params,
            activation=obs_act_activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            use_fc_ln=use_fc_ln,
            last_use_fc_bn=use_fc_bn,
            last_use_fc_ln=use_fc_ln,
            last_layer_activation=obs_act_activation,
            use_naive_parallel_network=use_naive_parallel_network,
            name=name + ".ObsActEncoder")

        if tau_input_processors is None:
            tau_input_processors = CosineEmbeddingPreprocessor(
                tau_spec, tau_embedding_dim)

        tau_encoder = EncodingNetwork(
            tau_spec,
            output_tensor_spec=TensorSpec(
                (tau_spec.numel, obs_act_tau_joint_fc_layer_params[0])),
            input_preprocessors=tau_input_processors,
            fc_layer_params=tau_fc_layer_params,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            use_fc_ln=use_fc_ln,
            last_layer_size=obs_act_tau_joint_fc_layer_params[0],
            last_activation=tau_activation,
            last_use_fc_ln=use_fc_ln,
            last_kernel_initializer=last_kernel_initializer,
            name=name + ".TauEncoder")

        super().__init__(
            input_tensor_spec=(input_tensor_spec, tau_spec),
            output_tensor_spec=TensorSpec((tau_spec.numel, ) +
                                          output_tensor_spec.shape),
            input_preprocessors=(obs_act_encoder, tau_encoder),
            preprocessing_combiner=alf.layers.NestMultiply(),
            fc_layer_params=obs_act_tau_joint_fc_layer_params,
            kernel_initializer=kernel_initializer,
            last_layer_size=output_tensor_spec.numel,
            last_activation=math_ops.identity,
            use_fc_bn=use_fc_bn,
            use_fc_ln=use_fc_ln,
            last_kernel_initializer=last_kernel_initializer,
            name=name)

        self._use_naive_parallel_network = use_naive_parallel_network

    def make_parallel(self, n):
        """Create a parallel critic network using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        If ``use_naive_parallel_network`` is True, use ``NaiveParallelNetwork``
        to create the parallel network.
        """
        if self._use_naive_parallel_network:
            return alf.networks.NaiveParallelNetwork(self, n)
        else:
            return super().make_parallel(n, True)


@alf.configurable
class RBFCriticNetwork(Network):
    """Critic network using RBF (Radial Basis Function) encoding.

    Architecture:
    1. Concatenate observation and action
    2. RBFEncodingNetwork (gamma scaling → RBF layer → sine activation)
    3. Final projection to scalar Q-value
    """

    def __init__(self,
                 input_tensor_spec,
                 n_components: int = 1000,
                 gamma: float = 3.0,
                 only_sign_matters: bool = False,
                 last_kernel_initializer=None,
                 use_bias=False,
                 name="RBFCriticNetwork"):
        """
        Args:
            input_tensor_spec (tuple[TensorSpec]):
                (observation_spec, action_spec)
            n_components (int): number of RBF components
            gamma (float): RBF bandwidth parameter
            last_kernel_initializer (Callable): initializer for final layer.
                If None, defaults to Normal(0, sqrt(1/n_components))
            use_bias (bool): whether to use bias in the final layer
            name (str): name of the network
        """
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)
        self._only_sign_matters = only_sign_matters

        observation_spec, action_spec = input_tensor_spec
        input_dim = observation_spec.numel + action_spec.numel

        # Create concatenated input spec for RBFEncodingNetwork
        joint_spec = TensorSpec((input_dim, ))

        # Create RBF encoding network
        self._encoding_net = RBFEncodingNetwork(input_tensor_spec=joint_spec,
                                                n_components=n_components,
                                                gamma=gamma,
                                                name=name + ".rbf_encoder")

        # Final projection layer: RBF features → scalar Q-value
        if last_kernel_initializer is None:
            last_kernel_initializer = functools.partial(torch.nn.init.normal_)

        self._value_layer = layers.FC(
            n_components,
            1,
            activation=lambda x: x,
            kernel_initializer=last_kernel_initializer,
            use_bias=use_bias)

    def forward(self, observation_action, state=()):
        """
        Args:
            observation_action (tuple): (observation, action)
            state (tuple): empty tuple (for API consistency)

        Returns:
            tuple:
            - q_value (torch.Tensor): shape [batch_size]
            - state (tuple): empty tuple
        """
        observation, action = observation_action
        if self._only_sign_matters:
            if action.shape[-1] != 1:
                raise ValueError(
                    "only_sign_matters=True requires 1D continuous action")
            a = action[..., :1]
            a_neg = torch.full_like(a, -1.0)
            a_pos = torch.full_like(a, 1.0)
            obs2 = torch.cat([observation, observation], dim=0)
            act2 = torch.cat([a_neg, a_pos], dim=0)
            joint2 = torch.cat([obs2, act2], dim=-1)
            rbf_features2, _ = self._encoding_net(joint2, state)
            q2 = self._value_layer(rbf_features2).squeeze(-1)
            q_neg, q_pos = q2.chunk(2, dim=0)
            t = (a.squeeze(-1) + 1.0) * 0.5
            q_value = q_neg + t * (q_pos - q_neg)
            return q_value, state

        joint = torch.cat([observation, action], dim=-1)
        rbf_features, _ = self._encoding_net(joint, state)
        q_value = self._value_layer(rbf_features).squeeze(-1)
        return q_value, state

    def make_parallel(self, n):
        """Create a parallel critic network using n replicas.

        Uses NaiveParallelNetwork for simplicity. The RBF layer is relatively
        lightweight, so naive parallelization is sufficient.
        """
        return alf.networks.NaiveParallelNetwork(self, n)


@alf.configurable
class RandomizedPriorCriticNetwork(Network):
    """A CriticNetwork augmented with a randomized prior function.

    This network creates two instances of a critic network: one trainable
    network and one frozen prior network. The outputs are summed to give
    the final critic value. This implements the randomized prior functions
    technique for improved exploration in continuous control tasks.

    Similar to RandomizedPriorQNetwork but for continuous action critics.
    """

    def __init__(self,
                 network_ctor: Callable = CriticNetwork,
                 input_tensor_spec=None,
                 prior_scale: float = 1.0,
                 trainable_init_std: float = 1e-3,
                 name="RandomizedPriorCriticNetwork",
                 **network_kwargs):
        """Creates a CriticNetwork with randomized prior.

        Args:
            network_ctor: Constructor for the base critic network (e.g.,
                CriticNetwork).
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            prior_scale: Target standard deviation for the prior network's output.
                The weight initialization is scaled so the output has this std.
            trainable_init_std: Target standard deviation for the trainable
                network's output. Typically small (e.g., 1e-3) so the network
                starts near zero.
            name: Name of the network.
            **network_kwargs: Additional arguments passed to network_ctor.
        """
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        # Compute last layer input dimension to scale weights correctly
        # output_std = weight_std * sqrt(input_dim), so weight_std = output_std / sqrt(input_dim)
        temp_net = network_ctor(input_tensor_spec=input_tensor_spec,
                                **network_kwargs)
        last_layer_input_dim = self._get_last_layer_input_dim(temp_net)
        del temp_net

        # Scale weights so OUTPUT has desired std
        trainable_weight_std = 3 * trainable_init_std / math.sqrt(
            last_layer_input_dim)
        prior_weight_std = 3 * prior_scale / math.sqrt(last_layer_input_dim)

        # Trainable network with small initialization
        trainable_init = functools.partial(torch.nn.init.normal_,
                                           std=trainable_weight_std)
        self._trainable_net = network_ctor(
            input_tensor_spec=input_tensor_spec,
            last_kernel_initializer=trainable_init,
            **network_kwargs)

        # Prior network with larger random initialization
        prior_init = functools.partial(torch.nn.init.normal_,
                                       std=prior_weight_std)
        self._prior_net = network_ctor(input_tensor_spec=input_tensor_spec,
                                       last_kernel_initializer=prior_init,
                                       **network_kwargs)

        # Freeze the prior network
        for param in self._prior_net.parameters():
            param.requires_grad = False

    def _get_last_layer_input_dim(self, net):
        """Find the input dimension of the last FC layer."""
        last_fc = None
        for module in net.modules():
            if isinstance(module, layers.FC):
                last_fc = module
        return last_fc.weight.shape[1] if last_fc else 1

    def forward(self, observation, state=()):
        """Computes critic values by summing trainable network and prior.

        Args:
            observation: Tuple of (observation, action) consistent with input_tensor_spec.
            state: Network state (for RNN-based networks).

        Returns:
            tuple:
            - critic_value (torch.Tensor): Sum of trainable and prior critic values.
            - state: Updated state.
        """
        q_vals, state = self._trainable_net(observation, state)
        with torch.no_grad():
            prior_vals, _ = self._prior_net(observation, state)
        return q_vals + prior_vals, state

    @property
    def state_spec(self):
        """Return the state spec (delegates to trainable network)."""
        return self._trainable_net.state_spec

    def make_parallel(self, n):
        """Make both sub-networks parallel for better performance."""
        parallel_trainable = self._trainable_net.make_parallel(n)
        parallel_prior = self._prior_net.make_parallel(n)
        return _ParallelRandomizedPriorCriticNetwork(parallel_trainable,
                                                     parallel_prior,
                                                     self.input_tensor_spec)


class _ParallelRandomizedPriorCriticNetwork(Network):
    """Parallel version of RandomizedPriorCriticNetwork."""

    def __init__(self,
                 parallel_trainable,
                 parallel_prior,
                 input_tensor_spec,
                 name="ParallelRandomizedPriorCriticNetwork"):
        super().__init__(input_tensor_spec=input_tensor_spec, name=name)
        self._trainable_net = parallel_trainable
        self._prior_net = parallel_prior
        self._output_spec = parallel_trainable.output_spec

    def forward(self, observation, state=()):
        q_vals, state = self._trainable_net(observation, state)
        with torch.no_grad():
            prior_vals, _ = self._prior_net(observation, state)
        return q_vals + prior_vals, state

    @property
    def state_spec(self):
        return self._trainable_net.state_spec
