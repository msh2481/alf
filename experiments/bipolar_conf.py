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
from alf.algorithms.seed_sampling import SeedSacAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.networks import QNetwork, RandomizedPriorQNetwork, RandomizedPriorCriticNetwork, RBFCriticNetwork
from alf.networks.actor_distribution_networks import RBFActorDistributionNetwork
from alf.utils.losses import element_wise_squared_loss
from functools import partial
from alf.environments import suite_gym
from alf.algorithms.bipolar_callback import BipolarCallback

ENV = alf.define_config('env', "BipolarChain-medium-dense-onehot-discrete-v0")
LR = alf.define_config('lr', 0.05)
WD = alf.define_config('wd', 1e-4)
GAMMA = alf.define_config('gamma', 0.9)
PRIOR_SCALE = alf.define_config('prior_scale', 0.1)
ALPHA = alf.define_config('alpha', 5e-4)
TAU = alf.define_config('tau', 0.05)
UTD = alf.define_config('utd', 8)
NUM_AGENTS = alf.define_config('num_agents', 1)
ASYNC = alf.define_config('async', True)
ENTROPY_REWARD = alf.define_config('entropy_reward', False)
N_COMPONENTS = alf.define_config('n_components', 4000)

DISCRETE = "discrete" in ENV
NUM_COPIES = NUM_AGENTS
BATCH_SIZE = 256 * NUM_COPIES
ENV_COUNTS = NUM_COPIES
UNROLL_LENGTH = 1
MINI_BATCH_LENGTH = 2

HIDDEN_LAYERS = (256, 256)

alf.config('create_environment',
           env_name=ENV,
           num_parallel_environments=ENV_COUNTS,
           ensure_different_phases=ASYNC,
           max_steps_for_phase_randomization=24)

alf.config('ReplayBuffer', shuffle_batch=True)
alf.config('BipolarCallback', annotate_transition_counts=True)

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

    alf.config('RBFCriticNetwork',
               n_components=N_COMPONENTS,
               gamma=2.0,
               only_sign_matters=True)
    alf.config('RBFActorDistributionNetwork',
               n_components=N_COMPONENTS,
               gamma=10.0,
               continuous_projection_net_ctor=partial(
                   alf.networks.BetaProjectionNetwork, min_concentration=1.0))
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
    target_update_tau=TAU,
    target_update_period=1,
    use_entropy_reward=ENTROPY_REWARD,
    num_actor_updates=None,
    **sac_kwargs,
)

alf.config('OneStepTDLoss',
           td_error_loss_fn=element_wise_squared_loss,
           gamma=GAMMA)

alf.config('ConcurrentAlgorithm', agent_reset_period=1, log_states=False)

alf.config(
    "ConcurrentAlgorithm",
    algorithm_ctor=SacAlgorithm,
    prior_perturbation_alpha=ALPHA,
    optimizer=alf.optimizers.Adam(lr=LR, weight_decay=WD, name='main'),
    num_copies=NUM_COPIES,
    use_exploration_seeds=False,
    debug_env=suite_gym.load(ENV),
    debug_callback_cls=BipolarCallback,
    video_record_interval=None,
    debug_log_every_n_steps=5,
    log_episode_returns=True,
)

alf.config('TrainerConfig',
           algorithm_ctor=ConcurrentAlgorithm,
           initial_collect_steps=10,
           mini_batch_length=MINI_BATCH_LENGTH,
           mini_batch_size=BATCH_SIZE,
           unroll_length=UNROLL_LENGTH,
           num_updates_per_train_iter=UTD,
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
