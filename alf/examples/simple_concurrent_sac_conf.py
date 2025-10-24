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
"""Simple Concurrent SAC Demo Configuration.

This demonstrates SimpleConcurrentAlgorithm with SAC on CartPole-v0.
The algorithm creates multiple independent SAC copies that learn concurrently.
"""

import alf

from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.simple_concurrent_algorithm import SimpleConcurrentAlgorithm, lazy_partial
from alf.algorithms.agent import Agent
from alf.networks import QNetwork
from alf.utils.losses import element_wise_squared_loss
from functools import partial

# environment config
alf.config('create_environment',
           env_name="CartPole-v0",
           num_parallel_environments=4)

# algorithm config
alf.config('QNetwork', fc_layer_params=(100, ))

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.98)

alf.config('SacAlgorithm', q_network_cls=QNetwork, target_update_tau=0.01)

# Use lazy_partial with lambdas to avoid sharing optimizers between algorithm copies
# Each lambda is called during algorithm construction to create fresh optimizer instances
alf.config(
    "SimpleConcurrentAlgorithm",
    algorithm_ctor=partial(
        Agent,
        rl_algorithm_cls=lazy_partial(
            SacAlgorithm,
            actor_optimizer=lambda: alf.optimizers.Adam(lr=1e-3, name='actor'),
            critic_optimizer=lambda: alf.optimizers.Adam(lr=1e-3,
                                                         name='critic'),
            alpha_optimizer=lambda: alf.optimizers.Adam(lr=1e-3, name='alpha'),
        )),
    num_copies=1)

alf.config('TrainerConfig',
           algorithm_ctor=SimpleConcurrentAlgorithm,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False,
           initial_collect_steps=1000,
           mini_batch_length=2,
           mini_batch_size=64,
           unroll_length=1,
           num_updates_per_train_iter=1,
           num_iterations=10000,
           num_checkpoints=5,
           evaluate=False,
           eval_interval=100,
           debug_summaries=True,
           summary_interval=100,
           replay_buffer_length=100000)
