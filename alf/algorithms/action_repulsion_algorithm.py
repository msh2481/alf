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
from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.algorithms.bipolar_callback import BipolarCallback
from alf.data_structures import LossInfo
from alf.tensor_specs import TensorSpec
from itertools import combinations
from alf.utils.common import warning


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
        debug_env=None,
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
            debug_env=debug_env,
        )

        self._repulsion_alpha = repulsion_alpha
        self._repulsion_num_obs = repulsion_num_obs

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
