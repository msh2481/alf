# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import torch


class Rotator(gym.Env):

    def __init__(self,
                 dt: float = 0.01,
                 num_steps: int = 1000,
                 eps: float = 1e-8):
        super().__init__()
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.eps = float(eps)

        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(2, ),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(2, ),
                                       dtype=np.float32)

        self._state = np.zeros((2, ), dtype=np.float32)
        self._step_count = 0

    def reset(self):
        self._state[...] = (0.5, 0.0)
        self._step_count = 0
        return self._state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        s0, s1 = float(self._state[0]), float(self._state[1])
        a0, a1 = float(action[0]), float(action[1])
        cross = s0 * a1 - s1 * a0
        reward = cross / (float(np.linalg.norm(self._state)) + self.eps)

        self._state[:] = np.clip(self._state + action * self.dt, -1.0, 1.0)
        self._step_count += 1
        done = self._step_count >= self.num_steps
        return self._state.copy(), float(reward), bool(done), {}

    def _grid(self, grid_res: int, device: torch.device):
        xs = torch.linspace(-1.0, 1.0, grid_res, device=device)
        ys = torch.linspace(-1.0, 1.0, grid_res, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        obs = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(torch.float32)
        return xs, ys, obs

    def get_value_grid(self, q_batch_fn, grid_res: int = 41, device=None):
        device = torch.device("cpu") if device is None else device
        _, _, obs = self._grid(int(grid_res), device)
        actions = torch.zeros_like(obs)
        v = q_batch_fn(obs, actions).reshape(grid_res, grid_res)
        return v.detach().cpu().numpy().T

    def get_actor_grid(self, actor_batch_fn, grid_res: int = 41, device=None):
        device = torch.device("cpu") if device is None else device
        _, _, obs = self._grid(int(grid_res), device)
        a = actor_batch_fn(obs).reshape(grid_res, grid_res, 2)
        a = a.detach().cpu().numpy()
        return a[:, :, 0].T, a[:, :, 1].T
