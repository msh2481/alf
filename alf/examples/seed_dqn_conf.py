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

from alf.algorithms.seed_sampling import SeedDqnAlgorithm, SeedSacAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.dqn_algorithm import DqnAlgorithm
from alf.algorithms.action_repulsion_algorithm import ActionRepulsionAlgorithm
from alf.networks import QNetwork, QNetworkBase, DebugLinearQNetwork
from alf.networks.encoding_networks import IdentityEncodingNetwork, EncodingNetwork
from alf.utils.losses import element_wise_squared_loss

BATCH_SIZE = 64
ENV_COUNTS = 4
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2
NUM_COPIES = 4
SEED_VERSION = True

# environment config
alf.config(
    'create_environment',
    env_name="RandomizedBipolarChain-v0",
    # env_name="CartPole-v0",
    # env_name="Pendulum-v0",
    num_parallel_environments=ENV_COUNTS)

# alf.config('QNetworkBase', fc_layer_params=(97, ))
# alf.config('QNetworkBase', encoding_network_ctor=EncodingNetwork)
alf.config(
    'SeedDqnAlgorithm' if SEED_VERSION else 'DqnAlgorithm',
    rollout_epsilon_greedy=0.0,
    # q_network_cls=QNetworkBase,
    q_network_cls=DebugLinearQNetwork,
    num_critic_replicas=1,
    #    actor_optimizer=alf.optimizers.Adam(lr=1e-3, name='actor'),
    #    critic_optimizer=alf.optimizers.Adam(lr=1e-3, name='critic'),
    #    alpha_optimizer=alf.optimizers.Adam(lr=1e-3, name='alpha'),
    parameter_target_std=0.1,
    parameter_target_alpha=1e-9,
    reward_noise_std=0.0,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.9)

alf.config(
    "ActionRepulsionAlgorithm",
    algorithm_ctor=SeedDqnAlgorithm if SEED_VERSION else DqnAlgorithm,
    optimizer=alf.optimizers.Adam(lr=5e-3, name='main'),
    num_copies=NUM_COPIES,
    batch_size=BATCH_SIZE,
    env_counts=ENV_COUNTS,
    unroll_length=UNROLL_LENGTH,
    mini_batch_length=MINI_BATCH_LENGTH,
    # repulsion_alpha=1e-4,
    repulsion_num_obs=100,
    log_every_n_steps=500,
    use_exploration_seeds=SEED_VERSION,
)

# training config
alf.config('TrainerConfig',
           algorithm_ctor=ActionRepulsionAlgorithm,
           initial_collect_steps=10,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=4,
           num_iterations=10000,
           num_checkpoints=3,
           evaluate=False,
           eval_interval=100,
           debug_summaries=True,
           summary_interval=100,
           replay_buffer_length=100000,
           random_seed=42,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False)
