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


class CheckValue(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(1, ),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    def reset(self):
        return np.array([0.], dtype=np.float32)

    def step(self, action):
        return np.array([0.], dtype=np.float32), 1.0, True, {}

    def render(self, mode="human", close=False):
        pass
