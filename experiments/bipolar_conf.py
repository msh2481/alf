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
from alf.networks import QNetwork, QNetworkBase, DebugLinearQNetwork, RandomizedPriorQNetwork, OptimisticQNetwork
from alf.networks.encoding_networks import IdentityEncodingNetwork, EncodingNetwork
from alf.utils.losses import element_wise_squared_loss
from alf.environments.simple.bipolar_chain import BipolarChain
from alf.environments import suite_gym

ENV_NAME = "BipolarChain-medium-sparse-onehot-v0"
NUM_COPIES = 6
BATCH_SIZE = 100 * NUM_COPIES
ENV_COUNTS = NUM_COPIES
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2
SEED_VERSION = True

HIDDEN_LAYERS = (1024, )

PRIOR_SCALE = 10.0
PARAMETER_TARGET_STD = 0.0
PARAMETER_TARGET_ALPHA = 0.0
REWARD_NOISE_STD = 0.0
assert (PRIOR_SCALE == 0.0) or (
    PARAMETER_TARGET_STD == 0.0 and PARAMETER_TARGET_ALPHA == 0.0
), "Don't turn on both PRIOR_SCALE and PARAMETER_TARGET_STD/PARAMETER_TARGET_ALPHA"

# environment config
alf.config('create_environment',
           env_name=ENV_NAME,
           num_parallel_environments=ENV_COUNTS)

alf.config('QNetwork', fc_layer_params=HIDDEN_LAYERS)
alf.config('OptimisticQNetwork', init_mean=0.0, init_std=1e-3)
alf.config(
    'RandomizedPriorQNetwork',
    #    network_ctor=QNetwork,
    network_ctor=DebugLinearQNetwork,
    prior_scale=PRIOR_SCALE)

alf.config(
    'SeedSacAlgorithm' if SEED_VERSION else 'SacAlgorithm',
    use_entropy_reward=False,
    q_network_cls=RandomizedPriorQNetwork,
    parameter_target_std=PARAMETER_TARGET_STD,
    parameter_target_alpha=PARAMETER_TARGET_ALPHA,
    reward_noise_std=REWARD_NOISE_STD,
)
alf.config(
    'SacAlgorithm',
    num_critic_replicas=1,
    target_update_tau=0.05,
    target_update_period=1,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.9)

alf.config(
    "ActionRepulsionAlgorithm",
    algorithm_ctor=SeedSacAlgorithm if SEED_VERSION else SacAlgorithm,
    optimizer=alf.optimizers.Adam(lr=1e-2, name='main'),
    num_copies=NUM_COPIES,
    batch_size=BATCH_SIZE,
    env_counts=ENV_COUNTS,
    unroll_length=UNROLL_LENGTH,
    mini_batch_length=MINI_BATCH_LENGTH,
    log_every_n_steps=100,
    use_exploration_seeds=SEED_VERSION,
    debug_env=suite_gym.load(ENV_NAME),
    video_record_interval=None,
)

# training config
alf.config('TrainerConfig',
           algorithm_ctor=ActionRepulsionAlgorithm,
           initial_collect_steps=10,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=5,
           num_iterations=10000,
           num_checkpoints=3,
           evaluate=False,
           eval_interval=100,
           replay_buffer_length=20000,
           random_seed=42,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False,
           summarize_grads_and_vars=False,
           debug_summaries=False,
           summary_interval=100,
           summarize_first_interval=False)
