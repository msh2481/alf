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
from functools import partial

from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.networks import RandomizedPriorCriticNetwork, RBFCriticNetwork
from alf.networks.actor_distribution_networks import RBFActorDistributionNetwork
from alf.environments import suite_dmc
from alf.environments.gym_wrappers import FrameSkip
from alf.utils.math_ops import clipped_exp
from alf.utils.losses import element_wise_squared_loss

NUM_COPIES = 4
BATCH_SIZE = 256 * NUM_COPIES
ENV_COUNTS = NUM_COPIES
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2

HIDDEN_LAYERS = (256, )
N_COMPONENTS = 1000
PRIOR_SCALE = 0.1

# environment config
alf.config('create_environment',
           env_name="cartpole:swingup_sparse",
           env_load_fn=suite_dmc.load,
           num_parallel_environments=ENV_COUNTS,
           ensure_different_phases=True,
           max_steps_for_phase_randomization=500)

alf.config('suite_dmc.load',
           from_pixels=False,
           max_episode_steps=1000,
           gym_env_wrappers=(partial(FrameSkip, skip=2), ))

alf.config('RBFCriticNetwork',
           n_components=N_COMPONENTS,
           gamma=2.0,
           only_sign_matters=True)
alf.config('RBFActorDistributionNetwork',
           n_components=N_COMPONENTS,
           gamma=2.0,
           continuous_projection_net_ctor=partial(
               alf.networks.NormalProjectionNetwork,
               state_dependent_std=True,
               std_transform=clipped_exp,
               scale_distribution=True,
               use_bias=False))
alf.config('RandomizedPriorCriticNetwork',
           network_ctor=RBFCriticNetwork,
           prior_scale=PRIOR_SCALE,
           trainable_init_std=1e-3)

alf.config(
    'SacAlgorithm',
    actor_network_cls=RBFActorDistributionNetwork,
    critic_network_cls=RandomizedPriorCriticNetwork,
    target_update_tau=0.005,
    target_update_period=1,
    use_entropy_reward=True,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.998)

alf.config('ConcurrentAlgorithm',
           log_states=True,
           agent_reset_period=1000,
           algorithm_ctor=SacAlgorithm,
           optimizer=alf.optimizers.AdamW(lr=0.05,
                                          weight_decay=1e-4,
                                          name='main'),
           num_copies=NUM_COPIES,
           use_exploration_seeds=False,
           return_logging_interval=500,
           video_record_interval=10000)

# training config
alf.config('TrainerConfig',
           algorithm_ctor=ConcurrentAlgorithm,
           initial_collect_steps=100,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=1,
           num_iterations=400000,
           num_checkpoints=3,
           evaluate=False,
           debug_summaries=False,
           summary_interval=200,
           replay_buffer_length=100000,
           random_seed=42,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False)
