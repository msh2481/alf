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


class LongHorizon(gym.Env):

    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(1, ),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(1, ),
                                            dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        obs = np.array([self.step_count / self.T], dtype=np.float32)
        reward = 1.0 if self.step_count > 0.9 * self.T else 0.0
        done = self.step_count >= self.T
        return obs, reward, done, {}

    def get_q_curves(self, q_callable, n_points=500, actions=(-0.5, 0, 0.5)):
        obs_values = np.linspace(0, 1, n_points, dtype=np.float32)
        results = {a: np.zeros(n_points, dtype=np.float32) for a in actions}
        for i, obs_val in enumerate(obs_values):
            obs = torch.tensor([obs_val], dtype=torch.float32)
            for a in actions:
                act = torch.tensor([a], dtype=torch.float32)
                results[a][i] = q_callable(obs, act)
        return obs_values, results
