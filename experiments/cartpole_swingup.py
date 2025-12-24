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

from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.networks import RandomizedPriorCriticNetwork, CriticNetwork
from alf.environments import suite_dmc
from alf.environments.gym_wrappers import FrameSkip
from alf.utils.math_ops import clipped_exp
from alf.utils.losses import element_wise_squared_loss

# Configurable hyperparameters (can be overridden via --conf_param)
LR = alf.define_config('lr', 1e-3)
WD = alf.define_config('wd', 0.1)
PRIOR_SCALE = alf.define_config('prior_scale', 0.1)
LN = alf.define_config('ln', True)
UTD = alf.define_config('utd', 1)
NUM_AGENTS = alf.define_config('num_agents', 1)
TAU = alf.define_config('tau', 0.1)
ASYNC = alf.define_config('async', True)

alf.config('create_environment',
           env_name="cartpole:swingup",
           env_load_fn=suite_dmc.load,
           num_parallel_environments=NUM_AGENTS,
           ensure_different_phases=ASYNC,
           max_steps_for_phase_randomization=125)

alf.config('suite_dmc.load',
           from_pixels=False,
           max_episode_steps=125,
           gym_env_wrappers=(partial(FrameSkip, skip=8), ))

alf.config('ActorDistributionNetwork',
           fc_layer_params=(256, ),
           continuous_projection_net_ctor=partial(
               alf.networks.NormalProjectionNetwork,
               state_dependent_std=True,
               scale_distribution=True,
               std_transform=clipped_exp))

alf.config('CriticNetwork', joint_fc_layer_params=(256, ), use_fc_ln=LN)

alf.config('RandomizedPriorCriticNetwork',
           network_ctor=CriticNetwork,
           prior_scale=PRIOR_SCALE,
           trainable_init_std=1e-3)

alf.config(
    'SacAlgorithm',
    actor_network_cls=alf.networks.ActorDistributionNetwork,
    critic_network_cls=RandomizedPriorCriticNetwork,
    target_update_tau=TAU,
    target_update_period=1,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.99)

alf.config(
    "ConcurrentAlgorithm",
    algorithm_ctor=SacAlgorithm,
    agent_reset_period=10**9,
    optimizer=alf.optimizers.Adam(lr=LR, weight_decay=WD, name='main'),
    num_copies=NUM_AGENTS,
    return_logging_interval=500,
    video_record_interval=5000,
    log_states=True,
)

alf.config('TrainerConfig',
           algorithm_ctor=ConcurrentAlgorithm,
           initial_collect_steps=100,
           mini_batch_length=2,
           mini_batch_size=256 * NUM_AGENTS,
           unroll_length=1,
           num_updates_per_train_iter=UTD,
           num_iterations=50000,
           num_checkpoints=3,
           evaluate=False,
           debug_summaries=False,
           summary_interval=200,
           replay_buffer_length=100000,
           random_seed=42,
           whole_replay_buffer_training=False,
           clear_replay_buffer=False)
