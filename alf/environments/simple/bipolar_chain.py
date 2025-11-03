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

import gym
from gym import spaces
import numpy as np


class BipolarChain(gym.Env):

    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.num_states = 2 * k + 1
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.num_states, ),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state = 0
        self.step_count = 0

    def reset(self):
        self.state = 0
        self.step_count = 0
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[self.state + self.k] = 1.0
        return obs

    def step(self, action):
        self.step_count += 1
        done = abs(self.state) >= self.k or self.step_count >= 2 * self.k
        if not done:
            self.state += 2 * action - 1
        reward = 1.0 if (self.state == self.k and not done) else 0.0
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[self.state + self.k] = 1.0
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        pass
