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
import os
import torch
import alf
from absl import logging
import matplotlib.pyplot as plt
import matplotlib
from alf.algorithms.config import TrainerConfig
from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.data_structures import LossInfo
from alf.tensor_specs import TensorSpec
from itertools import combinations
from alf.utils.common import warning
import numpy as np


@alf.configurable
class ActionRepulsionAlgorithm(ConcurrentAlgorithm):
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
        log_every_n_steps: int = 100,
        env_class=None,
        video_record_interval: int | None = None,
        return_logging_interval: int = 100,
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
            video_record_interval=video_record_interval,
            return_logging_interval=return_logging_interval,
        )

        self._repulsion_alpha = repulsion_alpha
        self._repulsion_num_obs = repulsion_num_obs
        self._debug_count = 0
        self._log_every_n_steps = log_every_n_steps
        self._debug_env = env_class() if env_class is not None else None

        # Validate that all sub-algorithms have actor networks when using repulsion
        if self._repulsion_alpha > 0:
            for i, alg in enumerate(self._algorithms):
                assert alg._actor_network is not None, (
                    f"ActionRepulsionAlgorithm with repulsion_alpha > 0 requires all "
                    f"sub-algorithms to have actor networks. Algorithm {i} has no actor. "
                    f"For discrete actions, set use_discrete_actor=True in algorithm_ctor."
                )

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

    def get_action_distribution_params(self, alg_index: int,
                                       observations: torch.Tensor):
        """Get action distribution parameters from actor network.

        Args:
            alg_index: index of the algorithm
            observations: [B, obs_dim] tensor of observations

        Returns:
            tensor: [B, param_dim] distribution parameters
                - For discrete: probabilities [B, num_actions]
                - For continuous: concatenated [mean, stddev] [B, action_dim * 2]
        """
        assert 0 <= alg_index < self._num_copies, (
            f"alg_index {alg_index} out of range [0, {self._num_copies})")

        batch_size = observations.shape[0]
        alg = self._algorithms[alg_index]

        # Require actor network - no Q-value fallback
        assert alg._actor_network is not None, (
            f"Algorithm {alg_index} has no actor network. "
            "ActionRepulsionAlgorithm requires use_discrete_actor=True for discrete actions."
        )

        # Get action distribution from actor network
        action_dist, _ = alg._actor_network(observations, state=())

        if self._action_spec.is_discrete:
            # For discrete: extract probabilities
            # action_dist should be Categorical
            dist_params = action_dist.probs  # [B, num_actions]
        else:
            # For continuous: extract mean and stddev
            if hasattr(action_dist, 'base_dist'):
                # For transformed distributions (e.g., TanhNormal)
                base = action_dist.base_dist
            else:
                base = action_dist

            assert hasattr(base, 'mean') and hasattr(base, 'stddev'), (
                f"Expected distribution with mean and stddev, got {type(base)}"
            )

            # Concatenate mean and std: [B, action_dim * 2]
            dist_params = torch.cat([base.mean, base.stddev], dim=-1)

        assert dist_params.shape[0] == batch_size, (
            f"Batch size mismatch: expected {batch_size}, got {dist_params.shape[0]}"
        )

        return dist_params

    def _get_q_values(self, alg_index: int, observations: torch.Tensor,
                      actions: torch.Tensor):
        """Get Q-values for debugging purposes.

        Args:
            alg_index: index of the algorithm
            observations: [B, obs_dim] tensor of observations
            actions: [B, ...] tensor of actions

        Returns:
            q_values: [B] tensor of Q-values
        """
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

    def _get_policy_vector(self, alg_index: int, observations: torch.Tensor):
        """Get policy vector by concatenating distribution params across observations.

        Args:
            alg_index: index of the algorithm
            observations: [B, obs_dim] - multiple observation states

        Returns:
            policy_vector: [B * dist_param_dim] - flattened vector representing
                the policy's behavior across all sampled states
        """
        # Get distribution parameters for all observations: [B, param_dim]
        dist_params = self.get_action_distribution_params(
            alg_index, observations)

        # Flatten to create policy vector: [B * param_dim]
        policy_vector = dist_params.reshape(-1)

        return policy_vector

    def _get_action_repulsion_loss(self, observations: torch.Tensor):
        """Compute repulsion loss based on action distribution similarity.

        Args:
            observations: [B, obs_dim] sampled observations

        Returns:
            loss: scalar tensor (negative sum of pairwise distances)
        """
        policy_vectors = []
        for i in range(self._num_copies):
            policy_vectors.append(self._get_policy_vector(i, observations))

        total_distance = torch.zeros((), device=observations.device)

        if self._debug_summaries:
            distance_matrix = torch.zeros((self._num_copies, self._num_copies),
                                          dtype=torch.float32,
                                          device=observations.device)

        for i, j in combinations(range(self._num_copies), 2):
            # L2 distance between policy vectors
            distance = torch.norm(policy_vectors[i] - policy_vectors[j], p=2)
            total_distance = total_distance + distance

            if self._debug_summaries:
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        if self._debug_summaries and self._debug_count % self._log_every_n_steps == 1:
            print(f"Policy distance matrix (L2):")
            print(distance_matrix.cpu().numpy())

        # Negative because we want to maximize distance
        loss = -total_distance
        return loss

    def calc_loss(self, info) -> LossInfo:
        loss_info = super().calc_loss(info)
        if self._repulsion_alpha > 0:
            observations, _, _ = self.sample_state_action_distribution(
                num_samples=self._repulsion_num_obs)
            if observations is None:
                warning("No data in replay buffer")
                return loss_info
            repulsion_loss = self._get_action_repulsion_loss(observations)
            total_loss = loss_info.loss + self._repulsion_alpha * repulsion_loss
            loss_info = loss_info._replace(loss=total_loss)
        return loss_info

    def debug_metrics(self,
                      observations,
                      actions,
                      rewards=None,
                      iter_number=None):
        self._debug_count += 1
        if self._debug_count % self._log_every_n_steps != 1:
            return

        os.makedirs('logs', exist_ok=True)
        if iter_number is None:
            iter_number = self._debug_count
        log_file_path = f'logs/{iter_number}.txt'

        with open(log_file_path, 'w') as f:
            f.write("\n=== Action Repulsion Debug Metrics ===\n")

            if observations is None or actions is None:
                f.write("No data in replay buffer\n")
                f.write("=" * 40 + "\n")
                logging.info(f"Written debug metrics to {log_file_path}")
                return

            assert self._debug_env is not None, "env_class is None - must be passed to ActionRepulsionAlgorithm"
            assert hasattr(self._debug_env, 'get_q_value_table'), \
                f"Environment {type(self._debug_env)} missing get_q_value_table method"
            assert hasattr(self._debug_env, 'get_transition_counts_table'), \
                f"Environment {type(self._debug_env)} missing get_transition_counts_table method"

            device = alf.get_default_device()
            k = self._debug_env.k
            positions = list(range(-k, k + 1))

            fig, axes = plt.subplots(self._num_copies + 1,
                                     2,
                                     figsize=(24, 6 * (self._num_copies + 1)))

            transition_counts = self._debug_env.get_transition_counts_table(
                self._replay_buffer)

            for action_idx, action_name in enumerate(['Left', 'Right']):
                ax_t = axes[0, action_idx]
                data_t = transition_counts[:, :, action_idx].T
                im_t = ax_t.imshow(data_t,
                                   aspect='auto',
                                   cmap='Blues',
                                   origin='lower',
                                   vmin=0,
                                   vmax=5)
                ax_t.set_xlabel('Position')
                ax_t.set_ylabel('Time')
                ax_t.set_title(f'Transitions {action_name}')
                ax_t.set_xticks(range(0, 2 * k + 1, max(1, (2 * k + 1) // 8)))
                ax_t.set_xticklabels([
                    positions[j]
                    for j in range(0, 2 * k + 1, max(1, (2 * k + 1) // 8))
                ])
                ax_t.set_yticks(range(k + 1))
                plt.colorbar(im_t, ax=ax_t)

                for pos_idx in range(2 * k + 1):
                    for time_idx in range(k + 1):
                        value = data_t[time_idx, pos_idx]
                        ax_t.text(
                            pos_idx,
                            time_idx,
                            f"{int(value) if not np.isnan(value) else ''}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=6)

            for i in range(self._num_copies):

                def q_func(obs, action, alg_index=i):
                    obs = obs.to(device)
                    action = action.to(device)
                    return self._get_q_values(alg_index, obs.unsqueeze(0),
                                              action.unsqueeze(0))[0].item()

                q_values = self._debug_env.get_q_value_table(q_func)

                for action_idx, action_name in enumerate(['Left', 'Right']):
                    ax_q = axes[i + 1, action_idx]
                    data_q = q_values[:, :, action_idx].T
                    im_q = ax_q.imshow(data_q,
                                       aspect='auto',
                                       cmap='viridis',
                                       origin='lower',
                                       vmin=-0.05,
                                       vmax=0.05)
                    ax_q.set_xlabel('Position')
                    ax_q.set_ylabel('Time')
                    ax_q.set_title(f'Algorithm {i} Q-values {action_name}')
                    ax_q.set_xticks(
                        range(0, 2 * k + 1, max(1, (2 * k + 1) // 8)))
                    ax_q.set_xticklabels([
                        positions[j]
                        for j in range(0, 2 * k + 1, max(1, (2 * k + 1) // 8))
                    ])
                    ax_q.set_yticks(range(k + 1))
                    plt.colorbar(im_q, ax=ax_q)

                    for pos_idx in range(2 * k + 1):
                        for time_idx in range(k + 1):
                            value = data_q[time_idx, pos_idx]
                            ax_q.text(pos_idx,
                                      time_idx,
                                      f"{value:.3f}",
                                      ha="center",
                                      va="center",
                                      color="white",
                                      fontsize=6)

            plt.tight_layout()
            plot_path = f'logs/{iter_number}.png'
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logging.info(f"Written plot to {plot_path}")

            num_samples = observations.shape[0]
            replay_buffer = self._replay_buffer
            total_size = replay_buffer.total_size.item(
            ) if replay_buffer is not None else 0
            f.write(f"Sampled {num_samples} out of {total_size} experiences\n")

            if rewards is not None:
                mean_reward = rewards.mean().item()
                f.write(f"\nMean reward: {mean_reward:.4f}\n")

            if observations.dim() > 1:
                obs_flat = observations.reshape(observations.shape[0], -1)
            else:
                obs_flat = observations.unsqueeze(-1)
            mean = obs_flat.mean(dim=0)
            std = obs_flat.std(dim=0)
            f.write(f"\nObservation statistics:\n")
            mean_str = ', '.join([f"{val:.2f}" for val in mean])
            std_str = ', '.join([f"{val:.2f}" for val in std])
            f.write(f" Mean: [{mean_str}]\n")
            f.write(f"  Std: [{std_str}]\n")

            f.write(f"\nVisited actions statistics:\n")
            if self._action_spec.is_discrete:
                unique_actions, counts = torch.unique(actions,
                                                      return_counts=True,
                                                      dim=0)
                sorted_indices = torch.argsort(unique_actions, dim=0)
                unique_actions = unique_actions[sorted_indices]
                counts = counts[sorted_indices]

                f.write(f"  Unique actions and their counts:\n")
                for action, count in zip(unique_actions, counts):
                    percentage = 100.0 * count.item() / num_samples
                    q_values = []
                    for i in range(self._num_copies):
                        q_val = self._get_q_values(i, observations[:1],
                                                   action.unsqueeze(0))
                        q_values.append(q_val[0].item())
                    q_str = ", ".join(
                        [f"Q{i}={q:.3f}" for i, q in enumerate(q_values)])
                    f.write(
                        f"    Action {action.item()}: {count.item()} ({percentage:.1f}%), [{q_str}]\n"
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
                    f.write(f"  Action dim {i}: mean = {mean[i].item():.4f}, "
                            f"std = {std[i].item():.4f}, "
                            f"min = {min_val[i].item():.4f}, "
                            f"max = {max_val[i].item():.4f}\n")

            for i, alg in enumerate(self._algorithms):
                f.write(f"\n=== Algorithm #{i} ===\n")
                if not hasattr(alg._critic_networks, '_networks'):
                    continue
                critics = alg._critic_networks._networks
                target_critics = alg._target_critic_networks._networks
                for j, (critic, target_critic) in enumerate(
                        zip(critics, target_critics)):
                    if hasattr(critic, '_log_parameters'):
                        f.write(f"  Critic #{j}:\n")
                        log_str = critic._log_parameters()
                        f.write(log_str + "\n")

            f.write("=" * 40 + "\n")

        logging.info(f"Written debug metrics to {log_file_path}")

    def _call_debug_metrics(self):
        if self._debug_env is None:
            return
        observations, actions, rewards = self.sample_state_action_distribution(
            num_samples=1000)
        self.debug_metrics(observations,
                           actions,
                           rewards,
                           iter_number=self._debug_count)

    def after_train_iter(self, inputs, info):
        super().after_train_iter(inputs, info)
        if self._env is not None:
            self._call_debug_metrics()
