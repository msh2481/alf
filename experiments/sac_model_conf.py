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
from functools import partial

import alf
import torch

assert (
    not torch.cuda.is_available()
), "CUDA is available; run with CUDA hidden (e.g. CUDA_VISIBLE_DEVICES='') or use a CPU-only PyTorch build."

from alf.algorithms.complex_maze_callback import ComplexMazeCallback
from alf.algorithms.concurrent_algorithm import ConcurrentAlgorithm
from alf.algorithms.data_transformer import RewardMaskByEnvId
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rotator_callback import RotatorCallback
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.sac_grad_algorithm import SacGradAlgorithm
from alf.algorithms.sac_v_algorithm import SacVAlgorithm
from alf.environments import suite_dmc, suite_gym
from alf.environments.gym_wrappers import FrameSkip
from alf.networks import CriticNetwork, RandomizedPriorCriticNetwork
from alf.networks.value_networks import RandomizedPriorValueNetwork, ValueNetwork
from alf.utils.losses import element_wise_squared_loss

# ── algo selection ─────────────────────────────────────────────────────────────
ALGO = alf.define_config("algo", "sac")  # 'sac' | 'sac_v' | 'sac_grad'
_ALGO_MAP = {
    "sac": SacAlgorithm,
    "sac_v": SacVAlgorithm,
    "sac_grad": SacGradAlgorithm
}
assert ALGO in _ALGO_MAP, f"algo must be one of {list(_ALGO_MAP)}, got '{ALGO}'"
ALGO_CLS = _ALGO_MAP[ALGO]

# ── shared hypers ──────────────────────────────────────────────────────────────
LR = alf.define_config("lr", 1e-3)
WD = alf.define_config("wd", 0.01)
GRAD_CLIP = alf.define_config("grad_clip", 10.0)
CLIP_BY_GLOBAL_NORM = alf.define_config("clip_by_global_norm", False)
GAMMA = alf.define_config("gamma", 0.99)
PRIOR_SCALE = alf.define_config("prior_scale", 0.1)
ALPHA = alf.define_config("alpha", 3e-4)
LN = alf.define_config("ln", True)
UTD = alf.define_config("utd", 1)
RESET_PERIOD = alf.define_config("reset_period", 1)
NUM_AGENTS = alf.define_config("num_agents", 1)
NUM_ENVS = alf.define_config("num_envs", 32)
SCALE_BATCH = alf.define_config("scale_batch", True)
UNROLL_LENGTH = alf.define_config("unroll_length", 0.25)
TAU = alf.define_config("tau", 0.1)
N_CRITICS = alf.define_config("n_critics", 2)
ASYNC = alf.define_config("async", True)
ENV = alf.define_config("env", "cartpole:swingup_sparse")
USE_BETA = alf.define_config("use_beta", True)
SHARE_ACTOR = alf.define_config("share_actor", False)
SHARE_CRITIC = alf.define_config("share_critic", False)
SHARED_CRITIC_MODE = alf.define_config("shared_critic_mode", "first")
SHUFFLE = alf.define_config("shuffle", False)
OWN_ROLLOUT_FRACTION = alf.define_config("own_rollout_fraction", -1.0)
NUM_LAYERS = alf.define_config("num_layers", 2)

# ── model-based knobs ──────────────────────────────────────────────────────────
MODEL_LOSS_WEIGHT = alf.define_config("model_loss_weight", 1.0)
GRAD_SYNC_WEIGHT = alf.define_config("grad_sync_weight", 1.0)
DYNAMICS_HIDDEN = alf.define_config("dynamics_hidden", (256, 256))
REWARD_HIDDEN = alf.define_config("reward_hidden", (256, 256))

MINI_BATCH_LENGTH = 2
assert NUM_LAYERS >= 1, f"num_layers={NUM_LAYERS} must be >= 1."
HIDDEN_LAYERS = (256, ) * int(NUM_LAYERS)

_SIMPLE_ENVS = {
    "Rotator": ("Rotator-v0", RotatorCallback),
    "ComplexMaze": ("ComplexMaze-v0", ComplexMazeCallback),
}
_SIMPLE_ENV = next(
    (v for key, v in _SIMPLE_ENVS.items()
     if isinstance(ENV, str) and ENV.startswith(key)),
    None,
)
_IS_SIMPLE = _SIMPLE_ENV is not None
_ENV_NAME = _SIMPLE_ENV[0] if _IS_SIMPLE else ENV
_ENV_LOAD_FN = suite_gym.load if _IS_SIMPLE else suite_dmc.load
_DEBUG_CALLBACK_CLS = _SIMPLE_ENV[1] if _IS_SIMPLE else None
VIDEO_RECORD_INTERVAL = 10000 if not _IS_SIMPLE else 10**9

alf.config(
    "create_environment",
    env_name=_ENV_NAME,
    env_load_fn=_ENV_LOAD_FN,
    num_parallel_environments=NUM_ENVS,
    ensure_different_phases=ASYNC,
    max_steps_for_phase_randomization=125,
)

alf.config("ReplayBuffer", shuffle_batch=SHUFFLE)

alf.config(
    "suite_dmc.load",
    from_pixels=False,
    max_episode_steps=125,
    gym_env_wrappers=(partial(FrameSkip, skip=8), ),
)

if USE_BETA:
    proj_net = partial(alf.networks.BetaProjectionNetwork,
                       min_concentration=1.0)
else:
    proj_net = partial(
        alf.networks.StableNormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        min_std=1e-2,
        max_std=2.0,
    )

alf.config(
    "ActorDistributionNetwork",
    fc_layer_params=HIDDEN_LAYERS,
    continuous_projection_net_ctor=proj_net,
)

alf.config("CriticNetwork", joint_fc_layer_params=HIDDEN_LAYERS, use_fc_ln=LN)

alf.config("ValueNetwork", fc_layer_params=HIDDEN_LAYERS, use_fc_ln=LN)

alf.config(
    "RandomizedPriorCriticNetwork",
    network_ctor=CriticNetwork,
    prior_scale=PRIOR_SCALE,
    trainable_init_std=1e-3,
)

alf.config(
    "RandomizedPriorValueNetwork",
    network_ctor=ValueNetwork,
    prior_scale=PRIOR_SCALE,
    trainable_init_std=1e-3,
)

alf.config("OneStepTDLoss",
           td_error_loss_fn=element_wise_squared_loss,
           gamma=GAMMA)

_common_sac_kwargs = dict(
    actor_network_cls=alf.networks.ActorDistributionNetwork,
    critic_network_cls=RandomizedPriorCriticNetwork,
    max_log_alpha=0.0,
    use_entropy_reward=False,
    num_critic_replicas=N_CRITICS,
    critic_loss_ctor=OneStepTDLoss,
    target_update_tau=TAU,
    target_update_period=1,
)

alf.config("SacAlgorithm", **_common_sac_kwargs)

alf.config(
    "SacVAlgorithm",
    **_common_sac_kwargs,
    value_network_cls=RandomizedPriorValueNetwork,
    dynamics_hidden=DYNAMICS_HIDDEN,
    reward_hidden=REWARD_HIDDEN,
    model_loss_weight=MODEL_LOSS_WEIGHT,
)

alf.config(
    "SacGradAlgorithm",
    **_common_sac_kwargs,
    dynamics_hidden=DYNAMICS_HIDDEN,
    reward_hidden=REWARD_HIDDEN,
    model_loss_weight=MODEL_LOSS_WEIGHT,
    grad_sync_weight=GRAD_SYNC_WEIGHT,
)

alf.config(
    "ConcurrentAlgorithm",
    algorithm_ctor=ALGO_CLS,
    agent_reset_period=RESET_PERIOD,
    prior_perturbation_alpha=ALPHA,
    optimizer=alf.optimizers.Adam(
        lr=LR,
        weight_decay=WD,
        name="main",
        gradient_clipping=GRAD_CLIP,
        clip_by_global_norm=CLIP_BY_GLOBAL_NORM,
    ),
    num_copies=NUM_AGENTS,
    share_actor_across_copies=SHARE_ACTOR,
    share_critic_across_copies=SHARE_CRITIC,
    shared_critic_mode=SHARED_CRITIC_MODE,
    own_rollout_fraction=OWN_ROLLOUT_FRACTION,
    return_logging_interval=500,
    video_record_interval=VIDEO_RECORD_INTERVAL,
    log_states=False,
    log_episode_returns=True,
    log_weight_norms=True,
    log_grad_norms=True,
    debug_env=suite_gym.load(_ENV_NAME) if _IS_SIMPLE else None,
    debug_callback_cls=None,
    debug_log_every_n_steps=10**9,
)

_BASE_MINI_BATCH_SIZE = 256
MINI_BATCH_SIZE = (_BASE_MINI_BATCH_SIZE *
                   NUM_AGENTS if SCALE_BATCH else _BASE_MINI_BATCH_SIZE)
assert (NUM_ENVS % NUM_AGENTS == 0
        ), f"num_envs={NUM_ENVS} must be divisible by num_agents={NUM_AGENTS}."
if not SCALE_BATCH:
    assert MINI_BATCH_SIZE % NUM_AGENTS == 0, (
        f"mini_batch_size={MINI_BATCH_SIZE} must be divisible by "
        f"num_agents={NUM_AGENTS} when scale_batch=False. "
        "Either set scale_batch=True or use a num_agents that divides 256.")

alf.config(
    "TrainerConfig",
    algorithm_ctor=ConcurrentAlgorithm,
    initial_collect_steps=1000,
    mini_batch_length=MINI_BATCH_LENGTH,
    mini_batch_size=MINI_BATCH_SIZE,
    unroll_length=UNROLL_LENGTH,
    num_updates_per_train_iter=UTD,
    num_iterations=50000,
    num_checkpoints=3,
    resume_from_checkpoint=False,
    clear_run_dirs_if_not_resuming=True,
    evaluate=False,
    debug_summaries=False,
    summary_interval=200,
    replay_buffer_length=100000,
    random_seed=42,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
)
