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
import torch
import gym
from gym import spaces


class RandomizedBipolarChain(gym.Env):
    """BipolarChain environment with pseudorandomized action effects.

    For each state, a random bit is generated (using a fixed seed for
    reproducibility). When this bit is 1, the effect of the action is
    flipped (XOR operation with the action).
    """

    def __init__(self, k=8):
        super().__init__()
        self.k = k
        self.num_states = (2 * self.k + 1) * (self.k + 1)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.num_states, ),
                                            dtype=np.float32)

        rng = np.random.RandomState(42)
        self.action_flip_bits = rng.randint(0, 2, size=2 * self.k + 1)

        self.state = 0
        self.step_count = 0

    def reset(self):
        self.state = 0
        self.step_count = 0
        return self._get_observation()

    def _state_to_idx(self, position, time_step):
        pos_idx = position + self.k
        time_idx = time_step
        return pos_idx * (self.k + 1) + time_idx

    def _get_observation(self):
        assert abs(
            self.state
        ) <= self.step_count, f"state: {self.state}, step_count: {self.step_count}"
        assert self.step_count <= self.k, f"step_count: {self.step_count}"
        obs = np.zeros(self.num_states, dtype=np.float32)
        flat_idx = self._state_to_idx(self.state, self.step_count)
        obs[flat_idx] = 1.0
        return obs

    def step(self, action):
        flip_bit = self.action_flip_bits[self.state + self.k]
        action = action ^ flip_bit

        self.step_count += 1
        self.state += 2 * action - 1
        done = abs(self.state) >= self.k or self.step_count >= self.k
        reward = float(self.state == self.k)
        obs = self._get_observation()
        return obs, reward, done, {}

    def get_q_value_table(self, q_function_callable):
        q_values = np.full((2 * self.k + 1, self.k + 1, 2),
                           np.nan,
                           dtype=np.float32)

        for position in range(-self.k, self.k + 1):
            for time_step in range(self.k + 1):
                if abs(position
                       ) > time_step or abs(position) % 2 != time_step % 2:
                    continue

                obs = np.zeros(self.num_states, dtype=np.float32)
                flat_idx = self._state_to_idx(position, time_step)
                obs[flat_idx] = 1.0
                obs_tensor = torch.from_numpy(obs)

                flip_bit = self.action_flip_bits[position + self.k]

                for action in range(2):
                    q_value = q_function_callable(obs_tensor,
                                                  torch.tensor(action))
                    decoded_action = action ^ flip_bit
                    q_values[position + self.k, time_step,
                             decoded_action] = q_value

        return q_values

    def get_transition_counts_table(self, replay_buffer):
        counts = np.zeros((self.num_states, 2), dtype=np.int64)

        if replay_buffer is not None and replay_buffer.total_size > 0:
            batch_size = replay_buffer.total_size.item()
            batch_info = replay_buffer._sample(batch_size=batch_size,
                                               batch_length=1)
            observations = replay_buffer.get_field('observation',
                                                   batch_info.env_ids,
                                                   batch_info.positions)
            actions = replay_buffer.get_field('action', batch_info.env_ids,
                                              batch_info.positions)

            for obs, action in zip(observations, actions):
                if isinstance(obs, torch.Tensor):
                    flat_idx = torch.argmax(obs).item()
                else:
                    flat_idx = np.argmax(obs)

                if isinstance(action, torch.Tensor):
                    action_val = action.item()
                else:
                    action_val = int(action)

                counts[flat_idx, action_val] += 1

        result = np.full((2 * self.k + 1, self.k + 1, 2),
                         np.nan,
                         dtype=np.float32)
        for position in range(-self.k, self.k + 1):
            for time_step in range(self.k + 1):
                if abs(position
                       ) > time_step or abs(position) % 2 != time_step % 2:
                    continue

                flat_idx = self._state_to_idx(position, time_step)
                flip_bit = self.action_flip_bits[position + self.k]
                result[position + self.k, time_step, 0] = counts[flat_idx,
                                                                 flip_bit]
                result[position + self.k, time_step, 1] = counts[flat_idx,
                                                                 1 ^ flip_bit]

        return result
