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


class BipolarChain(gym.Env):
    """BipolarChain environment with pseudorandomized action effects.

    For each state, a random bit is generated (using a fixed seed for
    reproducibility). When this bit is 1, the effect of the action is
    flipped (XOR operation with the action).
    """

    def __init__(self, k=12, dense=False, factored=False, continuous=False):
        super().__init__()
        self.k = k
        self.dense = dense
        self.factored = factored
        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=(1, ),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)

        if self.factored:
            # Two integers: position (-k to k), time_step (0 to k)
            self.observation_space = spaces.Box(low=np.array([-self.k, 0],
                                                             dtype=np.int32),
                                                high=np.array([self.k, self.k],
                                                              dtype=np.int32),
                                                dtype=np.int32)
        else:
            # One-hot encoding (current behavior)
            self.num_states = (2 * self.k + 1) * (self.k + 1)
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
        return self._state_to_observation(self.state, self.step_count)

    def _state_to_observation(self, position, time_step):
        assert abs(position) <= self.k
        assert 0 <= time_step <= self.k
        if self.factored:
            return np.array([position, time_step], dtype=np.int32)
        else:
            # One-hot encoding
            obs = np.zeros(self.num_states, dtype=np.float32)
            flat_idx = (position + self.k) * (self.k + 1) + time_step
            obs[flat_idx] = 1.0
            return obs

    def _observation_to_state(self, obs):
        if self.factored:
            return int(obs[0]), int(obs[1])
        else:
            if isinstance(obs, torch.Tensor):
                flat_idx = torch.argmax(obs).item()
            else:
                flat_idx = np.argmax(obs)
            time_step = flat_idx % (self.k + 1)
            pos_idx = flat_idx // (self.k + 1)
            position = pos_idx - self.k
            return position, time_step

    def step(self, action):
        if self.continuous:
            action_val = action[0] if isinstance(action, (np.ndarray,
                                                          list)) else action
            if isinstance(action_val, torch.Tensor):
                action_val = action_val.item()
            action_val = float(action_val)
            assert -1.0 <= action_val <= 1.0, f"Action {action_val} not in [-1, 1]"
            action = int(action_val >= 0)
        flip_bit = self.action_flip_bits[self.state + self.k]
        action = action ^ flip_bit

        prev_state = self.state
        self.step_count += 1
        self.state += 2 * action - 1
        done = abs(self.state) >= self.k or self.step_count >= self.k

        if self.dense:
            # Incremental reward: +1/k for moving right, -1/k for moving left
            state_change = self.state - prev_state
            reward = state_change / self.k
        else:
            # Sparse reward: only at goal
            reward = float(self.state == self.k)

        obs = self._state_to_observation(self.state, self.step_count)
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
                obs = self._state_to_observation(position, time_step)
                obs_tensor = torch.from_numpy(obs)
                flip_bit = self.action_flip_bits[position + self.k]

                for action in range(2):
                    if self.continuous:
                        action_tensor = torch.tensor([2 * action - 1.0],
                                                     dtype=torch.float32)
                    else:
                        action_tensor = torch.tensor(action)
                    q_value = q_function_callable(obs_tensor, action_tensor)
                    decoded_action = action ^ flip_bit
                    q_values[position + self.k, time_step,
                             decoded_action] = q_value

        return q_values

    def get_transition_counts_table(self, replay_buffer):
        counts = np.zeros((2 * self.k + 1, self.k + 1, 2), dtype=np.int64)

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
                position, time_step = self._observation_to_state(obs)
                if self.continuous:
                    if isinstance(action, torch.Tensor):
                        action_val = action.item() if action.numel(
                        ) == 1 else action[0].item()
                    elif isinstance(action, np.ndarray):
                        action_val = float(action[0] if len(action.shape) >
                                           0 else action)
                    else:
                        action_val = float(action)
                    assert -1.0 <= action_val <= 1.0, f"Action {action_val} not in [-1, 1]"
                    action_val = int(action_val >= 0)
                else:
                    action_val = int(action)
                flip_bit = self.action_flip_bits[position + self.k]
                decoded_action = action_val ^ flip_bit
                assert 0 <= decoded_action <= 1, f"Decoded action {decoded_action} not in [0, 1]"
                counts[position + self.k, time_step, decoded_action] += 1

        # Convert to result format with NaN for invalid states
        result = np.full((2 * self.k + 1, self.k + 1, 2),
                         np.nan,
                         dtype=np.float32)
        for position in range(-self.k, self.k + 1):
            for time_step in range(self.k + 1):
                if abs(position
                       ) > time_step or abs(position) % 2 != time_step % 2:
                    continue
                result[position + self.k, time_step,
                       0] = counts[position + self.k, time_step, 0]
                result[position + self.k, time_step,
                       1] = counts[position + self.k, time_step, 1]

        return result
