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
                 dt: float = 0.08,
                 num_steps: int = 125,
                 eps: float = 1e-8,
                 c_distraction: float = 0.0,
                 sparsity_threshold: float = -0.5):
        super().__init__()
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.eps = float(eps)
        self.c_distraction = float(c_distraction)
        self.sparsity_threshold = float(sparsity_threshold)
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0],
                                                         dtype=np.float32),
                                            high=np.array([1.0, 1.0, 1.0],
                                                          dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(2, ),
                                       dtype=np.float32)

        self._state = np.zeros((3, ), dtype=np.float32)
        self._step_count = 0

    def reset(self):
        self._state[...] = (0.5, 0.0, 0.0)
        self._step_count = 0
        return self._state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        state_xy = self._state[:2].copy()
        next_state_xy = np.clip(state_xy + action * self.dt, -1.0, 1.0)
        if self.dt > 0:
            eff_action = (next_state_xy - state_xy) / self.dt
        else:
            eff_action = np.zeros_like(action)

        s0, s1 = map(float, state_xy)
        a0, a1 = map(float, eff_action)
        cross = s0 * a1 - s1 * a0
        reward = cross / (float(np.linalg.norm(state_xy)) + self.eps)
        # hide reward when to the right of the threshold
        if s0 > self.sparsity_threshold:
            reward = 0.0
        # yet distracting part is always there
        reward += self.c_distraction * a1

        self._state[:2] = next_state_xy
        self._step_count += 1
        done = self._step_count >= self.num_steps
        denom = float(max(1, self.num_steps))
        t_norm = float(self._step_count) / denom
        self._state[2] = np.float32(np.clip(t_norm, 0.0, 1.0))
        return self._state.copy(), float(reward), bool(done), {}

    def _grid(self, grid_res: int, device: torch.device, t: float = 0.0):
        xs = torch.linspace(-1.0, 1.0, grid_res, device=device)
        ys = torch.linspace(-1.0, 1.0, grid_res, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        zz = torch.full_like(xx, float(t))
        obs = torch.stack([xx, yy, zz], dim=-1).reshape(-1,
                                                        3).to(torch.float32)
        return xs, ys, obs

    def get_value_grid(self,
                       q_batch_fn,
                       grid_res: int = 41,
                       device=None,
                       t: float = 0.0):
        device = torch.device("cpu") if device is None else device
        _, _, obs = self._grid(int(grid_res), device, t=t)
        actions = torch.zeros((obs.shape[0], 2),
                              device=device,
                              dtype=torch.float32)
        v = q_batch_fn(obs, actions).reshape(grid_res, grid_res)
        return v.detach().cpu().numpy()

    def get_actor_grid(self,
                       actor_batch_fn,
                       grid_res: int = 41,
                       device=None,
                       t: float = 0.0):
        device = torch.device("cpu") if device is None else device
        _, _, obs = self._grid(int(grid_res), device, t=t)
        a = actor_batch_fn(obs).reshape(grid_res, grid_res, 2)
        a = a.detach().cpu().numpy()
        return a[:, :, 0], a[:, :, 1]
