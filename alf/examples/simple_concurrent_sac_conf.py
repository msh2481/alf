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

This demonstrates SimpleConcurrentAlgorithm with SAC on LunarLander-v2.
The algorithm creates multiple independent SAC copies that learn concurrently.
"""

import alf
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.simple_concurrent_algorithm import SimpleConcurrentAlgorithm
from alf.networks import QNetwork
from alf.optimizers import AdamTF
from alf.utils.losses import element_wise_squared_loss

# Environment configuration
alf.config(
    "create_environment",
    env_name="LunarLander-v2",
    num_parallel_environments=2,  # Must be multiple of num_copies (2)
)

# Q-Network configuration
alf.config("QNetwork", fc_layer_params=(128, 128))

# SAC algorithm configuration
alf.config(
    "SacAlgorithm",
    q_network_cls=QNetwork,
    actor_optimizer=AdamTF(lr=1e-3, name="actor"),
    critic_optimizer=AdamTF(lr=1e-3, name="critic"),
    alpha_optimizer=AdamTF(lr=1e-3, name="alpha"),
    target_update_tau=0.01,
)

alf.config("OneStepTDLoss",
           td_error_loss_fn=element_wise_squared_loss,
           gamma=0.99)

# SimpleConcurrentAlgorithm configuration
alf.config("SimpleConcurrentAlgorithm",
           algorithm_ctor=SacAlgorithm,
           num_copies=2)  # 2 independent SAC copies

# Training configuration
alf.config(
    "TrainerConfig",
    algorithm_ctor=SimpleConcurrentAlgorithm,
    num_iterations=10000,
    unroll_length=1,
    mini_batch_length=2,
    mini_batch_size=64,
    num_updates_per_train_iter=1,
    initial_collect_steps=1000,
    replay_buffer_length=20000,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    evaluate=True,
    eval_interval=200,
    num_eval_episodes=5,
    debug_summaries=True,
    random_seed=42,
)
