# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from typing import Callable, Optional
import torch
import alf
from absl import logging
from alf.algorithms.config import TrainerConfig
from alf.algorithms.simple_concurrent_algorithm import SimpleConcurrentAlgorithm
from alf.data_structures import LossInfo
from alf.tensor_specs import TensorSpec
from itertools import combinations
from alf.utils.common import warning


@alf.configurable
class ActionRepulsionAlgorithm(SimpleConcurrentAlgorithm):
    """Action Repulsion Algorithm with multiple concurrent learners.

    Encourages diversity between sub-algorithms by penalizing similar Q-value
    landscapes. Currently designed primarily for discrete action spaces and a small number
    of sub-algorithms (loss computation complexity is quadratic in num_copies).
    """

    def __init__(
        self,
        observation_spec,
        action_spec,
        algorithm_ctor: Callable,
        num_copies: int = 2,
        reward_spec=TensorSpec(()),
        env=None,
        config: Optional[TrainerConfig] = None,
        checkpoint: Optional[str] = None,
        optimizer=None,
        debug_summaries: bool = False,
        name: str = "ActionRepulsionAlgorithm",
        batch_size=None,
        env_counts=None,
        unroll_length=None,
        mini_batch_length=None,
        repulsion_alpha: float = 0.0,
        repulsion_num_obs: int = 100,
        use_exploration_seeds: bool = True,
    ):
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            algorithm_ctor=algorithm_ctor,
            num_copies=num_copies,
            reward_spec=reward_spec,
            env=env,
            config=config,
            checkpoint=checkpoint,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name,
            batch_size=batch_size,
            env_counts=env_counts,
            unroll_length=unroll_length,
            mini_batch_length=mini_batch_length,
            use_exploration_seeds=use_exploration_seeds,
        )

        self._repulsion_alpha = repulsion_alpha
        self._repulsion_num_obs = repulsion_num_obs

        logging.info(
            f"ActionRepulsionAlgorithm instantiated with: "
            f"num_copies={num_copies}, repulsion_alpha={repulsion_alpha}, "
            f"repulsion_num_obs={repulsion_num_obs}, "
            f"use_exploration_seeds={use_exploration_seeds}, "
            f"batch_size={batch_size}, env_counts={env_counts}, "
            f"unroll_length={unroll_length}, mini_batch_length={mini_batch_length}"
        )

    def sample_state_action_distribution(self, num_samples: int):
        replay_buffer = self._replay_buffer
        if replay_buffer is None:
            warning("No replay buffer")
            return None, None, None
        if replay_buffer.total_size == 0:
            warning("Replay buffer is empty")
            return None, None, None
        num_samples = min(num_samples, replay_buffer.total_size.item())
        batch_info = replay_buffer._sample(batch_size=num_samples,
                                           batch_length=1)
        observations = replay_buffer.get_field('observation',
                                               batch_info.env_ids,
                                               batch_info.positions)
        actions = replay_buffer.get_field('action', batch_info.env_ids,
                                          batch_info.positions)
        rewards = replay_buffer.get_field('reward', batch_info.env_ids,
                                          batch_info.positions)
        return observations, actions, rewards

    def get_q_values(self, alg_index: int, observations: torch.Tensor,
                     actions: torch.Tensor):
        assert 0 <= alg_index < self._num_copies, (
            f"alg_index {alg_index} out of range [0, {self._num_copies})")

        batch_size = observations.shape[0]
        assert actions.shape[0] == batch_size, (
            f"Batch size mismatch: observations {observations.shape[0]} vs actions {actions.shape[0]}"
        )

        alg = self._algorithms[alg_index]

        if self._action_spec.is_discrete:
            q_values, _ = alg._compute_critics(alg._critic_networks,
                                               observations,
                                               None,
                                               critics_state=(),
                                               replica_min=True,
                                               apply_reward_weights=True)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            q_values, _ = alg._compute_critics(alg._critic_networks,
                                               observations,
                                               actions,
                                               critics_state=(),
                                               replica_min=True,
                                               apply_reward_weights=True)

        assert q_values.shape == (batch_size, ), (
            f"Output batch size mismatch: expected {batch_size}, got {q_values.shape}"
        )
        return q_values

    def _get_policy_vector(self, alg_index: int, observations: torch.Tensor,
                           actions: torch.Tensor):
        B = observations.shape[0]
        A = actions.shape[0]

        obs_expanded = observations.unsqueeze(1).expand(
            B, A, *observations.shape[1:])
        act_expanded = actions.unsqueeze(0).expand(B, A, *actions.shape[1:])

        obs_flat = obs_expanded.reshape(B * A, *observations.shape[1:])
        act_flat = act_expanded.reshape(B * A, *actions.shape[1:])

        q_values = self.get_q_values(alg_index, obs_flat, act_flat)
        q_values = q_values.reshape(B, A)
        mean = q_values.mean(dim=1, keepdim=True)
        std = q_values.std() + 1e-8
        policy_vector = ((q_values - mean) / (std + 1e-8)).reshape(B * A)
        return policy_vector

    def _get_action_repulsion_loss(self, observations: torch.Tensor,
                                   actions: torch.Tensor):
        policy_vectors = []
        for i in range(self._num_copies):
            policy_vectors.append(
                self._get_policy_vector(i, observations, actions))
        total_distance = torch.zeros(())
        distance_matrix = torch.zeros((self._num_copies, self._num_copies),
                                      dtype=torch.int32)
        for i, j in combinations(range(self._num_copies), 2):
            distance = torch.norm(policy_vectors[i] - policy_vectors[j], p=2)
            distance_matrix[i, j] = int(distance.item() * 100)
            total_distance = total_distance + distance.sum()
        # print(f"distance_matrix:\n{distance_matrix.numpy()}")
        loss = -total_distance
        return loss

    def calc_loss(self, info) -> LossInfo:
        loss_info = super().calc_loss(info)
        if self._repulsion_alpha > 0:
            observations, actions, _ = self.sample_state_action_distribution(
                num_samples=self._repulsion_num_obs)
            actions = torch.unique(actions, dim=0)
            if observations is None or actions is None:
                warning("No data in replay buffer")
                return loss_info
            repulsion_loss = self._get_action_repulsion_loss(
                observations, actions)
            # print("repulsion_loss:", repulsion_loss.item(), "x",
            #   self._repulsion_alpha)
            total_loss = loss_info.loss + self._repulsion_alpha * repulsion_loss
            loss_info = loss_info._replace(loss=total_loss)
        return loss_info

    def debug_metrics(self, observations, actions, rewards=None):
        if torch.rand(1).item() > 0.01:
            return

        print("\n=== Action Repulsion Debug Metrics ===")

        if observations is None or actions is None:
            print("No data in replay buffer")
            print("=" * 40 + "\n")
            return

        num_samples = observations.shape[0]

        if rewards is not None:
            mean_reward = rewards.mean().item()
            print(f"\nMean reward: {mean_reward:.4f}")

        print(f"\nVisited states statistics (n={num_samples}):")
        if observations.dim() > 1:
            obs_flat = observations.reshape(observations.shape[0], -1)
        else:
            obs_flat = observations.unsqueeze(-1)

        mean = obs_flat.mean(dim=0)
        std = obs_flat.std(dim=0)

        for i in range(min(obs_flat.shape[1], 20)):
            print(
                f"  Obs dim {i}: mean = {mean[i].item():.4f}, std = {std[i].item():.4f}"
            )

        if obs_flat.shape[1] > 20:
            print(f"  ... ({obs_flat.shape[1] - 20} more dimensions)")

        print(f"\nVisited actions statistics (n={num_samples}):")
        if self._action_spec.is_discrete:
            unique_actions, counts = torch.unique(actions,
                                                  return_counts=True,
                                                  dim=0)
            sorted_indices = torch.argsort(unique_actions, dim=0)
            unique_actions = unique_actions[sorted_indices]
            counts = counts[sorted_indices]

            print(f"  Unique actions and their counts:")
            for action, count in zip(unique_actions, counts):
                percentage = 100.0 * count.item() / num_samples
                q_values = []
                for i in range(self._num_copies):
                    q_val = self.get_q_values(i, observations[:1],
                                              action.unsqueeze(0))
                    q_values.append(q_val[0].item())
                q_str = ", ".join(
                    [f"Q{i}={q:.3f}" for i, q in enumerate(q_values)])
                print(
                    f"    Action {action.item()}: {count.item()} ({percentage:.1f}%), [{q_str}]"
                )
        else:
            if actions.dim() > 1:
                action_flat = actions.reshape(actions.shape[0], -1)
            else:
                action_flat = actions.unsqueeze(-1)

            mean = action_flat.mean(dim=0)
            std = action_flat.std(dim=0)
            min_val = action_flat.min(dim=0)[0]
            max_val = action_flat.max(dim=0)[0]

            for i in range(action_flat.shape[1]):
                print(f"  Action dim {i}: mean = {mean[i].item():.4f}, "
                      f"std = {std[i].item():.4f}, "
                      f"min = {min_val[i].item():.4f}, "
                      f"max = {max_val[i].item():.4f}")

        for i, alg in enumerate(self._algorithms):
            print(f"=== Algorithm #{i} ===")
            critics = alg._critic_networks._networks
            for j, critic in enumerate(critics):
                print(f"  Critic #{j}")
                critic._log_parameters()

        print("=" * 40 + "\n")

    def _call_debug_metrics(self):
        observations, actions, rewards = self.sample_state_action_distribution(
            num_samples=1000)
        self.debug_metrics(observations, actions, rewards)

    def after_train_iter(self, inputs, info):
        super().after_train_iter(inputs, info)
        if self._debug_summaries and self._env is not None:
            self._call_debug_metrics()
