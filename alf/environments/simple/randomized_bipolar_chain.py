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

import numpy as np
import pandas as pd
import torch
from alf.environments.simple.bipolar_chain import BipolarChain


class RandomizedBipolarChain(BipolarChain):
    """BipolarChain environment with pseudorandomized action effects.

    For each state, a random bit is generated (using a fixed seed for
    reproducibility). When this bit is 1, the effect of the action is
    flipped (XOR operation with the action).
    """

    def __init__(self, k=10):
        super().__init__(k)
        rng = np.random.RandomState(42)
        self.action_flip_bits = rng.randint(0, 2, size=self.num_states)

    def step(self, action):
        # Randomly flip the action, to prevent trivial exploration
        flip_bit = self.action_flip_bits[self.state + self.k]
        action = action ^ flip_bit

        self.step_count += 1
        done = abs(self.state) >= self.k or self.step_count >= self.k + 1
        if not done:
            self.state += 2 * action - 1
        reward = 1.0 if (self.state == self.k and not done) else 0.0
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[self.state + self.k] = 1.0
        return obs, reward, done, {}

    def get_q_value_table(self, q_function_callable):
        q_values_right = []
        q_values_left = []

        for state in range(-self.k, self.k + 1):
            obs = np.zeros(self.num_states, dtype=np.float32)
            obs[state + self.k] = 1.0
            obs_tensor = torch.from_numpy(obs)

            q_action_0 = q_function_callable(obs_tensor, torch.tensor(0))
            q_action_1 = q_function_callable(obs_tensor, torch.tensor(1))

            flip_bit = self.action_flip_bits[state + self.k]
            actual_right_action = 1 ^ flip_bit

            if actual_right_action == 0:
                q_values_right.append(q_action_0)
                q_values_left.append(q_action_1)
            else:
                q_values_right.append(q_action_1)
                q_values_left.append(q_action_0)

        states = list(range(-self.k, self.k + 1))
        return pd.DataFrame([q_values_right, q_values_left],
                            index=["Right (↑)", "Left (↓)"],
                            columns=states)

    def get_transition_counts_table(self, replay_buffer):
        counts_right = np.zeros(2 * self.k + 1, dtype=np.int64)
        counts_left = np.zeros(2 * self.k + 1, dtype=np.int64)

        if replay_buffer is None or replay_buffer.total_size == 0:
            states = list(range(-self.k, self.k + 1))
            return pd.DataFrame([counts_right, counts_left],
                                index=["Right (↑)", "Left (↓)"],
                                columns=states)

        batch_size = min(10000, replay_buffer.total_size.item())
        batch_info = replay_buffer._sample(batch_size=batch_size,
                                           batch_length=1)
        observations = replay_buffer.get_field('observation',
                                               batch_info.env_ids,
                                               batch_info.positions)
        actions = replay_buffer.get_field('action', batch_info.env_ids,
                                          batch_info.positions)

        for obs, action in zip(observations, actions):
            if isinstance(obs, torch.Tensor):
                state_idx = torch.argmax(obs).item()
            else:
                state_idx = np.argmax(obs)

            if isinstance(action, torch.Tensor):
                action_val = action.item()
            else:
                action_val = int(action)

            flip_bit = self.action_flip_bits[state_idx]
            decoded_action = action_val ^ flip_bit

            if decoded_action == 1:
                counts_right[state_idx] += 1
            else:
                counts_left[state_idx] += 1

        states = list(range(-self.k, self.k + 1))
        return pd.DataFrame([counts_right, counts_left],
                            index=["Right (↑)", "Left (↓)"],
                            columns=states)
