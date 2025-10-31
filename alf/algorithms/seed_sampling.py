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
from alf.data_structures import LossInfo, TimeStep
from alf.debug_logger import log
from alf.nest_formatter import format_nest
from alf.utils.common import seed_rand_nested
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

        for param_name, param in self.named_parameters():
            if param.requires_grad:
                buffer = torch.zeros_like(param)
                buffer.normal_(mean=0.0, std=parameter_target_std)
                self.register_buffer(
                    f"_param_target_{param_name.replace('.', '_')}", buffer)
                self._seed_sampling_parameter_targets[param_name] = buffer

    def _seed_sampling_calc_loss_addition(self, loss_info: LossInfo):
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
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 q_network_cls=QNetwork,
                 repr_alg_ctor: Optional[Callable] = None,
                 reward_weights=None,
                 train_eps_greedy=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 use_mc_return=False,
                 normalize_entropy_reward=False,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau: Union[float, Scheduler] = 0.05,
                 target_update_period: Union[int, Scheduler] = 1,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="SeedSacAlgorithm",
                 reward_noise_std: float = 0.0,
                 parameter_target_std: float = 0.0,
                 parameter_target_alpha: float = 0.0,
                 exploration_seed: int = 0):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         reward_spec=reward_spec,
                         actor_network_cls=actor_network_cls,
                         critic_network_cls=critic_network_cls,
                         q_network_cls=q_network_cls,
                         repr_alg_ctor=repr_alg_ctor,
                         reward_weights=reward_weights,
                         train_eps_greedy=train_eps_greedy,
                         epsilon_greedy=epsilon_greedy,
                         use_entropy_reward=use_entropy_reward,
                         use_mc_return=use_mc_return,
                         normalize_entropy_reward=normalize_entropy_reward,
                         calculate_priority=calculate_priority,
                         num_critic_replicas=num_critic_replicas,
                         env=env,
                         config=config,
                         critic_loss_ctor=critic_loss_ctor,
                         target_entropy=target_entropy,
                         prior_actor_ctor=prior_actor_ctor,
                         target_kld_per_dim=target_kld_per_dim,
                         initial_log_alpha=initial_log_alpha,
                         max_log_alpha=max_log_alpha,
                         target_update_tau=target_update_tau,
                         target_update_period=target_update_period,
                         parameter_reset_period=parameter_reset_period,
                         dqda_clipping=dqda_clipping,
                         actor_optimizer=actor_optimizer,
                         critic_optimizer=critic_optimizer,
                         alpha_optimizer=alpha_optimizer,
                         checkpoint=checkpoint,
                         debug_summaries=debug_summaries,
                         reproduce_locomotion=reproduce_locomotion,
                         name=name)
        self._seed_sampling_init(reward_noise_std=reward_noise_std,
                                 parameter_target_std=parameter_target_std,
                                 parameter_target_alpha=parameter_target_alpha,
                                 exploration_seed=exploration_seed)

        logging.info(f"SeedSacAlgorithm instantiated with: "
                     f"reward_noise_std={reward_noise_std}, "
                     f"parameter_target_std={parameter_target_std}, "
                     f"parameter_target_alpha={parameter_target_alpha}, "
                     f"exploration_seed={exploration_seed}, "
                     f"target_update_tau={target_update_tau}")

    def calc_loss(self, info):
        loss_info = super().calc_loss(info)
        return self._seed_sampling_calc_loss_addition(loss_info)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        inputs = self._seed_sampling_train_step_preprocessing(inputs)
        return super().train_step(inputs, state, rollout_info)
