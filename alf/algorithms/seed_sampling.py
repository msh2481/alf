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

import torch
import torch.distributions as td
from absl import logging

import alf
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.dqn_algorithm import DqnAlgorithm
from alf.data_structures import LossInfo, TimeStep
from alf.debug_logger import log
from alf.nest_formatter import format_nest
from alf.nest import seed_rand_nested
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from typing import Callable, Optional, Union
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, LossInfo
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.schedulers import Scheduler
from alf.debug_logger import log
from alf.nest_formatter import format_nest


class SeedSamplingMixin:

    def _seed_sampling_init(self,
                            reward_noise_std: float = 0.0,
                            parameter_target_std: float = 0.0,
                            parameter_target_alpha: float = 0.0,
                            exploration_seed: int = 0):
        self._seed_sampling_reward_noise_std = reward_noise_std
        self._seed_sampling_parameter_target_std = parameter_target_std
        self._seed_sampling_parameter_target_alpha = parameter_target_alpha
        self._seed_sampling_exploration_seed = exploration_seed
        self._seed_sampling_parameter_targets = {}

        parameter_noise_on = self._seed_sampling_parameter_target_std > 0 or self._seed_sampling_parameter_target_alpha > 0
        if parameter_noise_on:
            for param_name, param in self.named_parameters():
                if param.requires_grad:
                    buffer = torch.zeros_like(param)
                    buffer.normal_(mean=0.0,
                                   std=0.0 if "bias" in param_name else
                                   parameter_target_std)
                    self.register_buffer(
                        f"_param_target_{param_name.replace('.', '_')}",
                        buffer)
                    self._seed_sampling_parameter_targets[param_name] = buffer
                    # Initialize parameter data to equal the buffer value
                    param.data.copy_(buffer)

    def _seed_sampling_calc_loss_addition(self, loss_info: LossInfo):
        parameter_noise_on = self._seed_sampling_parameter_target_std > 0 or self._seed_sampling_parameter_target_alpha > 0
        if parameter_noise_on:
            reg_loss = 0.0
            for param_name, param in self.named_parameters():
                if param.requires_grad and param_name in self._seed_sampling_parameter_targets:
                    target = self._seed_sampling_parameter_targets[param_name]
                    mse = torch.nn.functional.mse_loss(param, target)
                    reg_loss = reg_loss + mse

            reg_loss = reg_loss * self._seed_sampling_parameter_target_alpha
            loss_info = loss_info._replace(loss=loss_info.loss + reg_loss)
        return loss_info

    def _seed_sampling_train_step_preprocessing(self, inputs: TimeStep):
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
                prev_action_i = alf.nest.map_structure(lambda x: x[i],
                                                       inputs.prev_action)
                seed_values[i] = seed_rand_nested(
                    (obs_i, prev_action_i),
                    seed=self._seed_sampling_exploration_seed)

            norm = td.Normal(0.0, 1.0)
            noise = self._seed_sampling_reward_noise_std * norm.icdf(
                seed_values)
            reward = inputs.reward + noise
            inputs = inputs._replace(reward=reward)

        return inputs


@alf.configurable
class SeedSacAlgorithm(SacAlgorithm, SeedSamplingMixin):

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 debug_summaries: bool = False,
                 name="SeedSacAlgorithm",
                 trace_path: str | None = None,
                 trace_every_n_updates: int = 1,
                 trace_max_rows_per_update: int | None = None,
                 reward_noise_std: float = 0.0,
                 parameter_target_std: float = 0.0,
                 parameter_target_alpha: float = 0.0,
                 exploration_seed: int = 0):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         reward_spec=reward_spec,
                         env=env,
                         config=config,
                         debug_summaries=debug_summaries,
                         trace_path=trace_path,
                         trace_every_n_updates=trace_every_n_updates,
                         trace_max_rows_per_update=trace_max_rows_per_update,
                         name=name)
        self._seed_sampling_init(reward_noise_std=reward_noise_std,
                                 parameter_target_std=parameter_target_std,
                                 parameter_target_alpha=parameter_target_alpha,
                                 exploration_seed=exploration_seed)

        logging.info(f"SeedSacAlgorithm instantiated with: "
                     f"reward_noise_std={reward_noise_std}, "
                     f"parameter_target_std={parameter_target_std}, "
                     f"parameter_target_alpha={parameter_target_alpha}, "
                     f"exploration_seed={exploration_seed}")

    def calc_loss(self, info):
        loss_info = super().calc_loss(info)
        return self._seed_sampling_calc_loss_addition(loss_info)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        inputs = self._seed_sampling_train_step_preprocessing(inputs)
        return super().train_step(inputs, state, rollout_info)


@alf.configurable
class SeedDqnAlgorithm(DqnAlgorithm, SeedSamplingMixin):

    def __init__(self,
                 observation_spec: alf.tensor_specs.NestedTensorSpec,
                 action_spec: alf.tensor_specs.BoundedTensorSpec,
                 reward_spec: TensorSpec = TensorSpec(()),
                 q_network_cls: Callable[..., QNetwork] = QNetwork,
                 q_optimizer: Optional[torch.optim.Optimizer] = None,
                 rollout_epsilon_greedy: Union[float, Scheduler] = 0.1,
                 target_net_target_action: bool = True,
                 num_critic_replicas: int = 2,
                 env=None,
                 config: Optional[TrainerConfig] = None,
                 critic_loss_ctor=None,
                 checkpoint=None,
                 debug_summaries: bool = False,
                 name: str = "SeedDqnAlgorithm",
                 reward_noise_std: float = 0.0,
                 parameter_target_std: float = 0.0,
                 parameter_target_alpha: float = 0.0,
                 exploration_seed: int = 0):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         reward_spec=reward_spec,
                         q_network_cls=q_network_cls,
                         q_optimizer=q_optimizer,
                         rollout_epsilon_greedy=rollout_epsilon_greedy,
                         target_net_target_action=target_net_target_action,
                         num_critic_replicas=num_critic_replicas,
                         env=env,
                         config=config,
                         critic_loss_ctor=critic_loss_ctor,
                         checkpoint=checkpoint,
                         debug_summaries=debug_summaries,
                         name=name)
        self._seed_sampling_init(reward_noise_std=reward_noise_std,
                                 parameter_target_std=parameter_target_std,
                                 parameter_target_alpha=parameter_target_alpha,
                                 exploration_seed=exploration_seed)

        logging.info(f"SeedDqnAlgorithm instantiated with: "
                     f"reward_noise_std={reward_noise_std}, "
                     f"parameter_target_std={parameter_target_std}, "
                     f"parameter_target_alpha={parameter_target_alpha}, "
                     f"exploration_seed={exploration_seed}")

    def calc_loss(self, info):
        loss_info = super().calc_loss(info)
        return self._seed_sampling_calc_loss_addition(loss_info)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        inputs = self._seed_sampling_train_step_preprocessing(inputs)
        return super().train_step(inputs, state, rollout_info)
