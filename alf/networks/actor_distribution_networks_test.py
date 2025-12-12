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
"""Tests for alf.networks.actor_distribution_networks."""

from alf.networks.actor_distribution_networks import ParallelActorDistributionNetwork
from absl.testing import parameterized
import functools
from absl import logging
import time
import torch
import torch.distributions as td

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorDistributionNetwork
from alf.networks import ActorDistributionRNNNetwork, RBFActorDistributionNetwork
from alf.networks import NormalProjectionNetwork, CategoricalProjectionNetwork, SimpleProjectionNetwork
from alf.utils.common import zero_tensor_from_nested_spec
from alf.nest.utils import NestConcat
from alf.utils.dist_utils import DistributionSpec
from alf.utils.math_ops import clipped_exp


class TestActorDistributionNetworks(parameterized.TestCase, alf.test.TestCase):

    def setUp(self):
        self._input_spec = [
            TensorSpec((3, 20, 20), torch.float32),
            TensorSpec((1, 20, 20), torch.float32)
        ]
        self._image = zero_tensor_from_nested_spec(self._input_spec,
                                                   batch_size=1)
        self._conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        self._fc_layer_params = (100, )
        self._input_preprocessors = [torch.tanh, None]
        self._preprocessing_combiner = NestConcat(dim=0)

    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(ActorDistributionRNNNetwork,
                                             lstm_hidden_size=lstm_hidden_size,
                                             actor_fc_layer_params=(64, 32))
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = [()]
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
            state.append(())
        else:
            network_ctor = ActorDistributionNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_discrete_actor_distribution(self, lstm_hidden_size):
        action_spec = TensorSpec((), torch.int32)
        network_ctor, state = self._init(lstm_hidden_size)

        # action_spec is not bounded
        self.assertRaises(AssertionError,
                          network_ctor,
                          self._input_spec,
                          action_spec,
                          conv_layer_params=self._conv_layer_params)

        action_spec = BoundedTensorSpec((), torch.int32)
        actor_dist_net = network_ctor(
            self._input_spec,
            action_spec,
            input_preprocessors=self._input_preprocessors,
            preprocessing_combiner=self._preprocessing_combiner,
            conv_layer_params=self._conv_layer_params)

        act_dist, _ = actor_dist_net(self._image, state)
        actions = act_dist.sample((100, ))

        self.assertTrue(
            isinstance(actor_dist_net.output_spec, DistributionSpec))

        # (num_samples, batch_size)
        self.assertEqual(actions.shape, (100, 1))

        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_continuous_actor_distribution(self, lstm_hidden_size):
        action_spec = BoundedTensorSpec((3, ), torch.float32)

        network_ctor, state = self._init(lstm_hidden_size)

        actor_dist_net = network_ctor(
            self._input_spec,
            action_spec,
            input_preprocessors=self._input_preprocessors,
            preprocessing_combiner=self._preprocessing_combiner,
            conv_layer_params=self._conv_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))
        act_dist, _ = actor_dist_net(self._image, state)
        actions = act_dist.sample((100, ))

        self.assertTrue(
            isinstance(actor_dist_net.output_spec, DistributionSpec))

        # (num_samples, batch_size, action_spec_shape)
        self.assertEqual(actions.shape, (100, 1) + action_spec.shape)

        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

    @parameterized.parameters(((200, 100), ), (None, ))
    def test_mixed_actor_distributions(self, lstm_hidden_size):
        action_spec = dict(discrete=BoundedTensorSpec((), dtype="int64"),
                           continuous=BoundedTensorSpec((3, )))

        network_ctor, state = self._init(lstm_hidden_size)

        actor_dist_net = network_ctor(
            self._input_spec,
            action_spec,
            input_preprocessors=self._input_preprocessors,
            preprocessing_combiner=self._preprocessing_combiner,
            conv_layer_params=self._conv_layer_params)

        act_dist, state = actor_dist_net(self._image, state)

        self.assertTrue(
            isinstance(actor_dist_net.output_spec["discrete"],
                       DistributionSpec))
        self.assertTrue(
            isinstance(actor_dist_net.output_spec["continuous"],
                       DistributionSpec))

        self.assertTrue(isinstance(act_dist["discrete"], td.Categorical))
        self.assertTrue(isinstance(act_dist["continuous"].base_dist,
                                   td.Normal))

        if lstm_hidden_size is None:
            self.assertEqual(state, ())
        else:
            self.assertEqual(len(alf.nest.flatten(state)),
                             2 * len(lstm_hidden_size))

    def test_make_parallel(self):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        network_ctor, _ = self._init(None)
        replicas = 4
        batch_size = 128

        # test continuous action
        action_spec = BoundedTensorSpec((3, ), torch.float32)
        actor_dist_net = network_ctor(
            obs_spec,
            action_spec,
            conv_layer_params=self._conv_layer_params,
            fc_layer_params=self._fc_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))

        pnet = actor_dist_net.make_parallel(replicas)
        self.assertTrue(isinstance(pnet, ParallelActorDistributionNetwork))
        self.assertEqual(pnet.name, "parallel_" + actor_dist_net.name)
        self.assertTrue(
            isinstance(actor_dist_net.output_spec, DistributionSpec))
        act_dist, _ = pnet(obs_spec.randn((batch_size, )))
        actions = act_dist.sample()
        self.assertEqual(actions.shape,
                         (batch_size, replicas) + action_spec.shape)
        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

        # test discrete action

        action_spec = TensorSpec((), torch.int32)
        # action_spec is not bounded
        self.assertRaises(AttributeError,
                          network_ctor,
                          obs_spec,
                          action_spec,
                          conv_layer_params=self._conv_layer_params)

        action_spec = BoundedTensorSpec((), torch.int32)
        actor_dist_net = network_ctor(
            obs_spec, action_spec, conv_layer_params=self._conv_layer_params)

        pnet = actor_dist_net.make_parallel(replicas)
        act_dist, _ = pnet(obs_spec.randn((batch_size, )))
        actions = act_dist.sample()
        self.assertEqual(actions.shape,
                         (batch_size, replicas) + action_spec.shape)
        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

    def test_rnn_make_parallel(self):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        network_ctor, state = self._init(100)
        replicas = 2

        action_spec = BoundedTensorSpec((3, ), torch.float32)
        actor_dist_net = network_ctor(
            obs_spec,
            action_spec,
            conv_layer_params=self._conv_layer_params,
            fc_layer_params=self._fc_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))
        pnet = actor_dist_net.make_parallel(replicas)
        state = alf.layers.make_parallel_input(state, replicas)
        self.assertTrue(isinstance(pnet, ParallelActorDistributionNetwork))
        self.assertEqual(pnet.name, "parallel_" + actor_dist_net.name)
        self.assertEqual(
            pnet.state_spec,
            alf.nest.map_structure(
                functools.partial(TensorSpec.from_tensor, from_dim=1), state))

        act_dist, _ = pnet(obs_spec.randn((1, replicas)), state)
        actions = act_dist.sample()
        self.assertEqual(actions.shape, (1, replicas) + action_spec.shape)

    def test_generalization(self):
        torch.manual_seed(0)
        obs_dim = 31
        obs_spec = TensorSpec((obs_dim, ), torch.float32)
        action_spec = BoundedTensorSpec((1, ),
                                        torch.float32,
                                        minimum=-1.0,
                                        maximum=1.0)
        actor = RBFActorDistributionNetwork(
            obs_spec,
            action_spec,
            n_components=2000,
            gamma=5,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork,
                state_dependent_std=True,
                scale_distribution=True,
                std_transform=clipped_exp,
                use_bias=False),
        )
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.2, momentum=0.5)
        # optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)

        num_steps = 100
        num_test_inputs = 30
        results = []

        embeddings = torch.zeros(num_test_inputs, obs_dim)
        for i in range(num_test_inputs):
            for j in range(obs_dim):
                embeddings[i, j] = torch.exp(-torch.tensor(
                    (i - j)**2, dtype=torch.float32))

        emb_0 = embeddings[5:6]
        target = torch.tensor([[1.0]])

        for _ in range(num_steps):
            optimizer.zero_grad()
            act_dist, _ = actor(emb_0)
            loss = ((act_dist.rsample((100, )).mean() - target)**2).mean()
            loss.backward()
            optimizer.step()

            test_means = []
            with torch.no_grad():
                for i in range(num_test_inputs):
                    emb_i = embeddings[i:i + 1]
                    act_dist, _ = actor(emb_i)
                    test_means.append(act_dist.rsample((100, )).mean().item())
            results.append(test_means)

        from matplotlib import pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        x = list(range(num_test_inputs))
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots(figsize=(12, 8))
        for step_idx, values in enumerate(results):
            color = cmap(step_idx / num_steps)
            ax.plot(x, values, color=color, alpha=0.7)
        ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel('Input index i')
        ax.set_ylabel('Action mean')
        ax.set_title(
            'Generalization test: mean(emb(i)) after training on emb(5)')
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=num_steps))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Iteration')
        plt.show()


if __name__ == "__main__":
    alf.test.main()
