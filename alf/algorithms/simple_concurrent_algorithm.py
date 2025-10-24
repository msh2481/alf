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
"""Simple Concurrent RL Algorithm.

Creates multiple independent algorithm copies that train concurrently.
"""

from typing import Callable, Optional

import torch.nn as nn

import alf
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.tensor_specs import TensorSpec


@alf.configurable
class SimpleConcurrentAlgorithm(RLAlgorithm):
    """SimpleConcurrent Algorithm.

    Creates K independent copies of a base algorithm. Each copy maintains
    independent parameters, optimizers, replay buffer, and environment interactions.
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
        debug_summaries: bool = False,
        name: str = "SimpleConcurrentAlgorithm",
    ):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            algorithm_ctor (Callable): Function to construct the base algorithm.
                Will be called as ``algorithm_ctor(observation_spec, action_spec, ...)``.
            num_copies (int): Number of independent algorithm copies (K).
            reward_spec (TensorSpec): representing the reward(s).
            env (Environment): The batched environment. Each algorithm gets its own env.
            config (TrainerConfig): config for training.
            debug_summaries (bool): whether to create debug summaries.
            name (str): name of this algorithm.
        """
        self._num_copies = num_copies
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._reward_spec = reward_spec

        # Create K independent algorithm copies, each with its own environment
        algorithms = [
            algorithm_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                env=env,
                config=config,
                debug_summaries=debug_summaries,
                name=f"{name}_copy_{i}",
            ) for i in range(num_copies)
        ]

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=algorithms[0].train_state_spec,
            reward_spec=reward_spec,
            predict_state_spec=algorithms[0].predict_state_spec,
            rollout_state_spec=algorithms[0].rollout_state_spec,
            is_on_policy=algorithms[0].on_policy,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
        )
        self._algorithms = nn.ModuleList(algorithms)

    def train_iter(self):
        """Perform one training iteration for all algorithm copies.

        Each algorithm runs its own complete training iteration independently:
        - Unrolls in its own environment
        - Stores experiences in its own replay buffer
        - Samples and trains from its own replay buffer

        Returns:
            int: total number of samples trained on across all algorithms
        """
        total_steps = 0
        for alg in self._algorithms:
            steps = alg.train_iter()
            total_steps += steps
        return total_steps

    def load_offline_replay_buffer(self, untransformed_observation_spec,
                                   ddp_rank):
        """Load offline replay buffer for all sub-algorithms."""
        for alg in self._algorithms:
            alg.load_offline_replay_buffer(untransformed_observation_spec,
                                           ddp_rank)

    def get_step_metrics(self):
        """Get step metrics from the first algorithm."""
        return self._algorithms[0].get_step_metrics()

    def get_metrics(self):
        """Get metrics from the first algorithm."""
        return self._algorithms[0].get_metrics()

    def load_checkpoint(self, *args, **kwargs):
        """Load checkpoint for all sub-algorithms."""
        for alg in self._algorithms:
            if hasattr(alg, 'load_checkpoint'):
                alg.load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        """Save checkpoint for all sub-algorithms."""
        for alg in self._algorithms:
            if hasattr(alg, 'save_checkpoint'):
                alg.save_checkpoint(*args, **kwargs)

    def finish_train(self):
        """Finish training for all sub-algorithms."""
        for alg in self._algorithms:
            if hasattr(alg, 'finish_train'):
                alg.finish_train()

    def reset_state(self):
        """Reset state for all sub-algorithms."""
        for alg in self._algorithms:
            if hasattr(alg, 'reset_state'):
                alg.reset_state()

    def evaluate(self):
        """Evaluate all sub-algorithms."""
        for alg in self._algorithms:
            if hasattr(alg, 'evaluate'):
                alg.evaluate()

    def eval_uncertainty(self):
        """Eval uncertainty for all sub-algorithms."""
        for alg in self._algorithms:
            if hasattr(alg, 'eval_uncertainty'):
                alg.eval_uncertainty()

    def compute_paras_statistics(self):
        """Compute parameter statistics from first algorithm."""
        return self._algorithms[0].compute_paras_statistics()

    def get_optimizer_info(self):
        """Get optimizer info from first algorithm."""
        return self._algorithms[0].get_optimizer_info()

    def get_unoptimized_parameter_info(self):
        """Get unoptimized parameter info from first algorithm."""
        return self._algorithms[0].get_unoptimized_parameter_info()
