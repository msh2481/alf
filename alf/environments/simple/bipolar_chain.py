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

    def __init__(self, k=100):
        super().__init__()
        self.k = k
        self.observation_space = spaces.Box(low=-k,
                                            high=k,
                                            shape=(1, ),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state = 0

    def reset(self):
        self.state = 0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        delta = 2 * action - 1
        self.state += delta

        if self.state >= self.k:
            return np.array([self.state], dtype=np.float32), 1.0, True, {}
        elif self.state <= -self.k:
            return np.array([self.state], dtype=np.float32), 0.0, True, {}
        else:
            return np.array([self.state], dtype=np.float32), 0.0, False, {}

    def render(self, mode="human", close=False):
        pass
