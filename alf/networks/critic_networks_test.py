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
"""Tests for alf.networks.value_networks."""

from absl.testing import parameterized
from absl import logging
import functools
import time
import torch

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import CriticNetwork, CriticRNNNetwork, RandomizedPriorCriticNetwork
from alf.networks.network import NaiveParallelNetwork
from alf.networks.network_test import test_net_copy
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.nest.utils import NestConcat


class CriticNetworksTest(parameterized.TestCase, alf.test.TestCase):

    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            post_rnn_fc_layer_params = (6, 4)
            network_ctor = functools.partial(
                CriticRNNNetwork,
                lstm_hidden_size=lstm_hidden_size,
                critic_fc_layer_params=post_rnn_fc_layer_params)
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = []
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
        else:
            network_ctor = CriticNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_critic(self, lstm_hidden_size):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        action_spec = TensorSpec((5, ), torch.float32)
        input_spec = (obs_spec, action_spec)

        observation_conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        action_fc_layer_params = (10, 8)
        joint_fc_layer_params = (6, 4)

        image = obs_spec.zeros(outer_dims=(2, ))
        action = action_spec.randn(outer_dims=(2, ))

        network_input = (image, action)

        network_ctor, state = self._init(lstm_hidden_size)

        critic_net = network_ctor(
            input_spec,
            observation_conv_layer_params=observation_conv_layer_params,
            action_fc_layer_params=action_fc_layer_params,
            joint_fc_layer_params=joint_fc_layer_params)
        test_net_copy(critic_net)

        value, state = critic_net._test_forward()
        self.assertEqual(value.shape, (2, ))
        if lstm_hidden_size is None:
            self.assertEqual(state, ())

        value, state = critic_net(network_input, state)

        self.assertEqual(critic_net.output_spec, TensorSpec(()))
        # (batch_size,)
        self.assertEqual(value.shape, (2, ))

        # test make_parallel
        pnet = critic_net.make_parallel(6)
        test_net_copy(pnet)

        if lstm_hidden_size is not None:
            # shape of state should be [B, n, ...]
            self.assertRaises(AssertionError, pnet, network_input, state)

        state = alf.nest.map_structure(
            lambda x: x.unsqueeze(1).expand(x.shape[0], 6, x.shape[1]), state)

        value, state = pnet(network_input, state)
        self.assertEqual(pnet.output_spec, TensorSpec((6, )))
        self.assertEqual(value.shape, (2, 6))

    def test_make_parallel(self):
        obs_spec = TensorSpec((20, ), torch.float32)
        action_spec = TensorSpec((5, ), torch.float32)
        critic_net = CriticNetwork((obs_spec, action_spec),
                                   joint_fc_layer_params=(256, 256))

        replicas = 4
        # ParallelCriticNetwork (PCN) is not always faster than NaiveParallelNetwork (NPN).
        # On my machine, for this particular example, with replicas=2,
        # PCN is faster when batch_size in (128, 256, ..., 2048)
        # NPN is faster when batch_size in (4096, 8192, 16384).
        # For a moderately large replicas (32), and smaller batch_size (128),
        # the speed difference is huge: PCN is 20 times faster than NPN.
        batch_size = 128

        def _train(pnet, name):
            t0 = time.time()
            optimizer = alf.optimizers.AdamTF(lr=1e-4)
            optimizer.add_param_group({'params': list(pnet.parameters())})
            for _ in range(100):
                obs = obs_spec.randn((batch_size, ))
                action = action_spec.randn((batch_size, ))
                values = pnet((obs, action))[0]
                target = torch.randn_like(values)
                cost = ((values - target)**2).sum()
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            logging.info("%s time=%s cost=%s" %
                         (name, time.time() - t0, float(cost)))

        pnet = critic_net.make_parallel(replicas)
        _train(pnet, "ParallelCriticNetwork")

        pnet = NaiveParallelNetwork(critic_net, replicas)
        _train(pnet, "NaiveParallelNetwork")

    @parameterized.parameters((CriticNetwork, ), (CriticRNNNetwork, ))
    def test_discrete_action(self, net_ctor):
        obs_spec = TensorSpec((20, ))
        action_spec = BoundedTensorSpec((), dtype='int64')

        # doesn't support discrete action spec ...
        self.assertRaises(AssertionError, net_ctor, (obs_spec, action_spec))

        # ... unless an preprocessor is specified
        net_ctor(
            (obs_spec, action_spec),
            action_input_processors=EmbeddingPreprocessor(action_spec,
                                                          embedding_dim=10))

    @parameterized.parameters((CriticNetwork, ), (CriticRNNNetwork, ))
    def test_mixed_actions(self, net_ctor):
        obs_spec = TensorSpec((20, ))
        action_spec = dict(x=BoundedTensorSpec((), dtype='int64'),
                           y=BoundedTensorSpec((3, )))

        input_preprocessors = dict(x=EmbeddingPreprocessor(action_spec['x'],
                                                           embedding_dim=10),
                                   y=None)

        net_ctor = functools.partial(
            net_ctor, action_input_processors=input_preprocessors)

        # doesn't support mixed actions
        self.assertRaises(AssertionError, net_ctor, (obs_spec, action_spec))

        # ... unless a combiner is specified
        net_ctor((obs_spec, action_spec),
                 action_preprocessing_combiner=NestConcat())

    def test_randomized_prior_critic(self):
        """Test RandomizedPriorCriticNetwork with parallel execution."""
        obs_spec = TensorSpec((8, ))
        action_spec = BoundedTensorSpec((2, ), minimum=-1, maximum=1)
        input_spec = (obs_spec, action_spec)

        prior_scale = 1.0
        trainable_init_std = 1e-3
        critic = RandomizedPriorCriticNetwork(
            input_tensor_spec=input_spec,
            prior_scale=prior_scale,
            trainable_init_std=trainable_init_std,
            joint_fc_layer_params=(256, ))

        # Test single network with N(0,1) inputs
        batch_size = 1000
        obs = torch.randn(batch_size, 8)
        action = torch.randn(batch_size, 2)
        output, _ = critic((obs, action))

        # Check output shape
        self.assertEqual(output.shape, (batch_size, ))

        # Check output statistics (approximate, due to randomness)
        # Output std should be close to prior_scale (within 50% tolerance)
        self.assertGreater(output.std().item(), prior_scale * 0.5)
        self.assertLess(output.std().item(), prior_scale * 2.0)

        # Make parallel
        n_replicas = 3
        parallel_critic = critic.make_parallel(n_replicas)
        output_parallel, _ = parallel_critic((obs, action))

        # Check parallel output shape
        self.assertEqual(output_parallel.shape, (batch_size, n_replicas))

        # Each replica should have similar statistics
        for i in range(n_replicas):
            replica_out = output_parallel[:, i]
            self.assertGreater(replica_out.std().item(), prior_scale * 0.5)
            self.assertLess(replica_out.std().item(), prior_scale * 2.0)

    def _create_onehot_data(self, num_inputs):
        X_train = torch.eye(num_inputs, dtype=torch.float32)
        signs = torch.randint(0, 2,
                              (num_inputs, ), dtype=torch.float32) * 2 - 1
        return X_train, signs

    def _create_dary_data(self, num_inputs, base=2):
        import math
        num_bits = math.ceil(math.log(num_inputs, base))
        X_train = torch.zeros((num_inputs, num_bits), dtype=torch.float32)
        for i in range(num_inputs):
            val = i
            for bit_idx in range(num_bits):
                X_train[i, bit_idx] = val % base
                val //= base
        signs = torch.randint(0, 2,
                              (num_inputs, ), dtype=torch.float32) * 2 - 1
        return X_train, signs

    def test_representation_learning(self):
        sub_ctor = functools.partial(CriticNetwork,
                                     joint_fc_layer_params=(32, 64, 512),
                                     use_fc_ln=True)
        # sub_ctor = functools.partial(CriticNetwork, joint_fc_layer_params=(256, 256,), use_fc_ln=True)
        critic_ctor = functools.partial(RandomizedPriorCriticNetwork,
                                        network_ctor=sub_ctor)
        num_trials = 20
        num_inputs = 64
        lr = 0.01
        max_steps = 10000
        batch_ratio = 0.1
        batch_size = int(num_inputs * batch_ratio)
        print("\n" + "=" * 60)
        print(f"Starting representation learning test")
        print(
            f"Num inputs: {num_inputs}, Batch size: {batch_size}, LR: {lr}, Max steps: {max_steps}"
        )
        print("=" * 60)
        convergence_steps = []
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}")
            # X_train, signs = self._create_dary_data(num_inputs, base=4)
            X_train, signs = self._create_onehot_data(num_inputs)
            obs_size = X_train.shape[1]
            obs_spec = TensorSpec((obs_size, ), torch.float32)
            action_spec = BoundedTensorSpec((1, ),
                                            torch.float32,
                                            minimum=-1.0,
                                            maximum=1.0)
            input_spec = (obs_spec, action_spec)
            critic = critic_ctor(input_tensor_spec=input_spec)
            optimizer = alf.optimizers.Adam(lr=lr)
            optimizer.add_param_group({'params': list(critic.parameters())})
            converged = False
            for step in range(max_steps):
                with torch.no_grad():
                    actions_pos = torch.ones(num_inputs,
                                             1,
                                             dtype=torch.float32)
                    actions_neg = -torch.ones(
                        num_inputs, 1, dtype=torch.float32)
                    q_pos, _ = critic((X_train, actions_pos))
                    q_neg, _ = critic((X_train, actions_neg))
                    comparisons = (q_pos > q_neg)
                    matches = (comparisons.flatten() == (signs > 0).flatten())
                    matches_all_signs = matches.all().item()
                    matches_count = matches.sum().item()
                if matches_all_signs:
                    converged = True
                    print(f"  Converged at step {step}")
                    convergence_steps.append(step)
                    break
                batch_indices = torch.randperm(num_inputs)[:batch_size]
                X_batch = X_train[batch_indices]
                signs_batch = signs[batch_indices].view(batch_size, 1)
                X_batch_paired = torch.cat([X_batch, X_batch], dim=0)
                signs_batch_paired = torch.cat([signs_batch, signs_batch],
                                               dim=0)
                actions_batch = torch.cat(
                    [torch.ones(batch_size, 1), -torch.ones(batch_size, 1)],
                    dim=0)
                targets_batch = (signs_batch_paired * actions_batch).flatten()

                q_values, _ = critic((X_batch_paired, actions_batch))
                assert q_values.shape == targets_batch.shape, f"q_values.shape: {q_values.shape}, targets_batch.shape: {targets_batch.shape}"
                loss = ((q_values - targets_batch)**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not converged:
                print(
                    f"  Trial {trial + 1} did not converge within {max_steps} steps"
                )
                convergence_steps.append(max_steps)
        print("\n" + "=" * 60)
        print("Representation learning test summary")
        print("=" * 60)
        print(f"Convergence steps across {num_trials} trials:")
        for i, steps in enumerate(convergence_steps):
            status = "✓" if steps < max_steps else "✗"
            print(f"  Trial {i+1}: {steps:5d} steps {status}")
        converged_trials = sum(1 for s in convergence_steps if s < max_steps)
        print(
            f"\nSuccessfully converged: {converged_trials}/{num_trials} trials"
        )
        if converged_trials > 0:
            converged_steps_only = [
                s for s in convergence_steps if s < max_steps
            ]
            sorted_steps = sorted(converged_steps_only)
            n = len(sorted_steps)
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            iqm_steps = sum(sorted_steps[q1_idx:q3_idx]) / (
                q3_idx - q1_idx) if q3_idx > q1_idx else sorted_steps[0]
            print(
                f"Interquartile mean steps (converged trials): {iqm_steps:.1f}"
            )
        print("=" * 60 + "\n")
        self.assertGreater(converged_trials, 0,
                           "At least one trial should converge")


if __name__ == "__main__":
    alf.test.main()
