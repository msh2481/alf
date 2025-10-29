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
"""Seed Sampling decorator for RL algorithms."""

import torch
import torch.distributions as td
import types
from typing import Callable, Optional

import alf
from alf.algorithms.config import TrainerConfig
from alf.data_structures import LossInfo, TimeStep
from alf.tensor_specs import TensorSpec
from alf.debug_logger import log
from alf.nest_formatter import format_nest
from alf.utils.common import seed_rand_nested


def _patched_calc_loss(self, info):
    original_calc_loss = self._seed_sampling_original_calc_loss
    loss_info = original_calc_loss(info)

    if self._seed_sampling_parameter_target_alpha > 0:
        reg_loss = 0.0
        for param_name, param in self.named_parameters():
            if param.requires_grad and param_name in self._seed_sampling_parameter_targets:
                target = self._seed_sampling_parameter_targets[param_name]
                mse = torch.nn.functional.mse_loss(param, target)
                reg_loss = reg_loss + mse

        reg_loss = reg_loss * self._seed_sampling_parameter_target_alpha
        loss_info = loss_info._replace(loss=loss_info.loss + reg_loss)

    log(f"loss_patched", format_nest(loss_info.loss))
    return loss_info


def _patched_train_step(self, inputs: TimeStep, state, rollout_info):
    if self._seed_sampling_reward_noise_std > 0:
        observation = inputs.observation
        first_tensor = next(iter(alf.nest.flatten(observation)))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        seed_values = torch.zeros(batch_size,
                                  device=device,
                                  dtype=torch.float32)
        for i in range(batch_size):
            obs_i = alf.nest.map_structure(lambda x: x[i], observation)
            seed_values[i] = seed_rand_nested(
                obs_i, seed=self._seed_sampling_exploration_seed)

        norm = td.Normal(0.0, 1.0)
        noise = self._seed_sampling_reward_noise_std * norm.icdf(seed_values)
        reward = inputs.reward + noise
        inputs = inputs._replace(reward=reward)

    return self._seed_sampling_original_train_step(inputs, state, rollout_info)


@alf.configurable
class SeedSampling:
    """A decorator that wraps a base RL algorithm.

    This class follows the decorator pattern, delegating all method calls to
    the base algorithm. The actual Seed Sampling logic will be implemented later
    by overriding specific methods.

    Args:
        algorithm_ctor: Constructor for the base RL algorithm (e.g., SacAlgorithm)
        observation_spec: Observation spec
        action_spec: Action spec
        reward_spec: Reward spec
        env: Environment
        config: Trainer config
        checkpoint: Checkpoint to load
        debug_summaries: Whether to create debug summaries
        name: Name of this algorithm
        reward_noise_std: Standard deviation for reward noise
        parameter_target_std: Target standard deviation for parameters
        parameter_target_alpha: Alpha parameter for target updates
        exploration_seed: Seed for exploration noise
    """

    def __init__(
        self,
        algorithm_ctor: Callable,
        observation_spec,
        action_spec,
        reward_spec=TensorSpec(()),
        env=None,
        config: Optional[TrainerConfig] = None,
        checkpoint: Optional[str] = None,
        debug_summaries: bool = False,
        name: str = "SeedSampling",
        reward_noise_std: float = 0.0,
        parameter_target_std: float = 0.0,
        parameter_target_alpha: float = 0.0,
        exploration_seed: int = 0,
    ):
        self._base_algorithm = algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=f"{name}_base",
        )

        self._base_algorithm._seed_sampling_original_calc_loss = self._base_algorithm.calc_loss
        self._base_algorithm._seed_sampling_original_train_step = self._base_algorithm.train_step
        self._base_algorithm._seed_sampling_reward_noise_std = reward_noise_std
        self._base_algorithm._seed_sampling_parameter_target_std = parameter_target_std
        self._base_algorithm._seed_sampling_parameter_target_alpha = parameter_target_alpha
        self._base_algorithm._seed_sampling_exploration_seed = exploration_seed
        self._base_algorithm._seed_sampling_parameter_targets = {}

        for param_name, param in self._base_algorithm.named_parameters():
            if param.requires_grad:
                buffer = torch.zeros_like(param)
                buffer.normal_(mean=0.0, std=parameter_target_std)
                self._base_algorithm.register_buffer(
                    f"_param_target_{param_name.replace('.', '_')}", buffer)
                self._base_algorithm._seed_sampling_parameter_targets[
                    param_name] = buffer

        self._base_algorithm.calc_loss = types.MethodType(
            _patched_calc_loss, self._base_algorithm)
        self._base_algorithm.train_step = types.MethodType(
            _patched_train_step, self._base_algorithm)

    def __getattr__(self, name):
        return object.__getattribute__(self._base_algorithm, name)
