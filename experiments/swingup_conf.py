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
from functools import partial

from alf.algorithms.seed_sampling import SeedSacAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.action_repulsion_algorithm import ActionRepulsionAlgorithm
from alf.environments import suite_dmc
from alf.environments.gym_wrappers import FrameSkip
from alf.utils.math_ops import clipped_exp
from alf.utils.losses import element_wise_squared_loss

NUM_COPIES = 4
BATCH_SIZE = 256 * NUM_COPIES
ENV_COUNTS = NUM_COPIES
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2
SEED_VERSION = True

# HIDDEN_LAYERS = (256, 256)
HIDDEN_LAYERS = (256, )

# environment config
alf.config('create_environment',
           env_name="cartpole:swingup",
           env_load_fn=suite_dmc.load,
           num_parallel_environments=ENV_COUNTS)

alf.config('suite_dmc.load',
           from_pixels=False,
           max_episode_steps=125,
           gym_env_wrappers=(partial(FrameSkip, skip=8), ))

# actor network for continuous actions
alf.config('ActorDistributionNetwork',
           fc_layer_params=HIDDEN_LAYERS,
           continuous_projection_net_ctor=partial(
               alf.networks.NormalProjectionNetwork,
               state_dependent_std=True,
               scale_distribution=True,
               std_transform=clipped_exp))

# critic network
alf.config('CriticNetwork', joint_fc_layer_params=HIDDEN_LAYERS)

alf.config(
    'SeedSacAlgorithm' if SEED_VERSION else 'SacAlgorithm',
    actor_network_cls=alf.networks.ActorDistributionNetwork,
    critic_network_cls=alf.networks.CriticNetwork,
    target_update_tau=0.005,
    target_update_period=1,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.99)

alf.config(
    "ActionRepulsionAlgorithm",
    algorithm_ctor=SeedSacAlgorithm if SEED_VERSION else SacAlgorithm,
    optimizer=alf.optimizers.Adam(lr=3e-4, name='main'),
    num_copies=NUM_COPIES,
    batch_size=BATCH_SIZE,
    env_counts=ENV_COUNTS,
    unroll_length=UNROLL_LENGTH,
    mini_batch_length=MINI_BATCH_LENGTH,
    log_every_n_steps=500,
    use_exploration_seeds=SEED_VERSION,
    video_record_interval=500,
)

# training config
alf.config('TrainerConfig',
           algorithm_ctor=ActionRepulsionAlgorithm,
           initial_collect_steps=1000,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=1,
           num_iterations=50000,
           num_checkpoints=3,
           evaluate=False,
           debug_summaries=False,
           summary_interval=200,
           replay_buffer_length=100000,
           random_seed=42,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False)
