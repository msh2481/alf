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
import alf

alf.import_config("sac_conf.py")
from alf.algorithms.seed_sampling import SeedSacAlgorithm
from alf.algorithms.action_repulsion_algorithm import ActionRepulsionAlgorithm
from alf.networks import QNetwork
from alf.utils.losses import element_wise_squared_loss

BATCH_SIZE = 64
ENV_COUNTS = 4
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2

# environment config
alf.config(
    'create_environment',
    # env_name="CheckPropagation-v0",
    env_name="CartPole-v0",
    # env_name="Pendulum-v0",
    num_parallel_environments=ENV_COUNTS)

alf.config('QNetwork', fc_layer_params=(97, ))
alf.config(
    'SeedSacAlgorithm',
    q_network_cls=QNetwork,
    #    actor_optimizer=alf.optimizers.Adam(lr=1e-3, name='actor'),
    #    critic_optimizer=alf.optimizers.Adam(lr=1e-3, name='critic'),
    #    alpha_optimizer=alf.optimizers.Adam(lr=1e-3, name='alpha'),
    target_update_tau=0.01,
    parameter_target_std=1.0,
    parameter_target_alpha=0.001,
    reward_noise_std=0.1,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.98)

alf.config(
    "ActionRepulsionAlgorithm",
    algorithm_ctor=SeedSacAlgorithm,
    optimizer=alf.optimizers.Adam(lr=1e-3, name='main'),
    num_copies=4,
    batch_size=BATCH_SIZE,
    env_counts=ENV_COUNTS,
    unroll_length=UNROLL_LENGTH,
    mini_batch_length=MINI_BATCH_LENGTH,
    repulsion_alpha=1e-10,
    repulsion_num_obs=100,
)

# training config
alf.config('TrainerConfig',
           algorithm_ctor=ActionRepulsionAlgorithm,
           initial_collect_steps=10,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=1,
           num_iterations=10000,
           num_checkpoints=3,
           evaluate=False,
           eval_interval=100,
           debug_summaries=True,
           summary_interval=100,
           replay_buffer_length=100000,
           random_seed=42)
