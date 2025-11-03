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


class ParallelChains(gym.Env):
    """Environment with k parallel chains of length l.

    State numbering:
    - State 0: initial state where agent chooses which chain to enter
    - State i*l + j (for i in [0, k-1], j in [1, l]): position j in chain i
    - Terminal states: when state % l == 0 (reached end of a chain)
    
    From state 0, action i selects chain i. From other states, any action
    advances to the next position in the current chain. Reward is 1.0 only
    when reaching the end of chain 0; all other chains give 0.0 reward.
    """

    def __init__(self, k=3, l=5):
        super().__init__()
        self.k = k
        self.l = l
        self.observation_space = spaces.Box(low=0,
                                            high=k * l,
                                            shape=(1, ),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(k)
        self.state = 0

    def reset(self):
        self.state = 0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        if self.state == 0:
            self.state = action * self.l + 1
        else:
            self.state += 1

        done = self.state % self.l == 0
        reward = 1.0 if done and self.state == self.l else 0.0
        return np.array([self.state], dtype=np.float32), reward, done, {}

    def render(self, mode="human", close=False):
        pass
