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
import math
import alf
from alf.algorithms.seed_sampling import SeedDqnAlgorithm, SeedSacAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.dqn_algorithm import DqnAlgorithm
from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.networks import QNetwork, RandomizedPriorQNetwork, CriticNetwork, RandomizedPriorCriticNetwork, RBFCriticNetwork
from alf.networks.actor_distribution_networks import RBFActorDistributionNetwork
from alf.networks.encoding_networks import IdentityEncodingNetwork, EncodingNetwork
from alf.utils.losses import element_wise_squared_loss
from alf.utils.math_ops import clipped_exp
from functools import partial
from alf.environments.simple.bipolar_chain import BipolarChain
from alf.environments import suite_gym

ENV_NAME = "BipolarChain-medium-sparse-onehot-continuous-v0"
DISCRETE = "discrete" in ENV_NAME
NUM_COPIES = 4
RESET_PERIOD = 200
BATCH_SIZE = 256 * NUM_COPIES
ENV_COUNTS = NUM_COPIES
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2
SEED_VERSION = False
ENTROPY_REWARD = False

HIDDEN_LAYERS = (256, 256)

PRIOR_SCALE = 5.0
# PARAMETER_TARGET_STD = 0.0
# PARAMETER_TARGET_ALPHA = 0.0
# REWARD_NOISE_STD = 0.0
# assert (PRIOR_SCALE == 0.0) or (
#     PARAMETER_TARGET_STD == 0.0 and PARAMETER_TARGET_ALPHA == 0.0
# ), "Don't turn on both PRIOR_SCALE and PARAMETER_TARGET_STD/PARAMETER_TARGET_ALPHA"

# environment config
alf.config('create_environment',
           env_name=ENV_NAME,
           num_parallel_environments=ENV_COUNTS)

alf.config('ReplayBuffer', shuffle_batch=True)

if DISCRETE:
    alf.config('QNetwork', fc_layer_params=HIDDEN_LAYERS, use_fc_ln=True)
    alf.config('RandomizedPriorQNetwork',
               network_ctor=QNetwork,
               prior_scale=PRIOR_SCALE)
    sac_kwargs = {
        'q_network_cls': RandomizedPriorQNetwork,
    }
else:
    # alf.config('ActorDistributionNetwork',
    #            fc_layer_params=HIDDEN_LAYERS,
    #            continuous_projection_net_ctor=partial(
    #                alf.networks.NormalProjectionNetwork,
    #                state_dependent_std=True,
    #                std_transform=clipped_exp,
    #                scale_distribution=True))
    # alf.config('CriticNetwork',
    #            joint_fc_layer_params=HIDDEN_LAYERS,
    #            use_fc_ln=True)

    N_COMPONENTS = 1000
    alf.config('RBFCriticNetwork',
               n_components=N_COMPONENTS,
               gamma=2.0,
               only_sign_matters=False)
    alf.config('RBFActorDistributionNetwork',
               n_components=N_COMPONENTS,
               gamma=10.0,
               continuous_projection_net_ctor=partial(
                   alf.networks.NormalProjectionNetwork,
                   zero_init=True,
                   state_dependent_std=True,
                   std_transform=partial(clipped_exp,
                                         clip_value_min=math.log(0.05),
                                         clip_value_max=math.log(0.2)),
                   scale_distribution=True,
                   use_bias=False))
    alf.config('RandomizedPriorCriticNetwork',
               network_ctor=RBFCriticNetwork,
               prior_scale=PRIOR_SCALE,
               trainable_init_std=1e-3)
    sac_kwargs = {
        'actor_network_cls': RBFActorDistributionNetwork,
        'critic_network_cls': RandomizedPriorCriticNetwork,
    }

alf.config(
    'SacAlgorithm',
    num_critic_replicas=1,
    target_update_tau=0.05,
    target_update_period=1,
    use_entropy_reward=ENTROPY_REWARD,
    **sac_kwargs,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.9)

alf.config('ConcurrentAlgorithm', agent_reset_period=RESET_PERIOD)

alf.config(
    "ConcurrentAlgorithm",
    algorithm_ctor=SeedSacAlgorithm if SEED_VERSION else SacAlgorithm,
    optimizer=alf.optimizers.AdamW(lr=1e-3, weight_decay=1e-2, name='main'),
    num_copies=NUM_COPIES,
    use_exploration_seeds=SEED_VERSION,
    debug_env=suite_gym.load(ENV_NAME),
    video_record_interval=None,
    debug_log_every_n_steps=10,
)

# training config
alf.config('TrainerConfig',
           algorithm_ctor=ConcurrentAlgorithm,
           initial_collect_steps=10,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=8,
           num_iterations=10000,
           num_checkpoints=3,
           evaluate=False,
           eval_interval=100,
           replay_buffer_length=20000,
           random_seed=0,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False,
           summarize_grads_and_vars=False,
           debug_summaries=False,
           summary_interval=100,
           summarize_first_interval=False)
