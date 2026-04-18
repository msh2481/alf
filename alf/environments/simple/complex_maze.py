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
"""ComplexMaze: a continuous exploration environment with GP-sampled dynamics.

State and action live in [0,1]^2, interpreted as complex numbers.
The transition is:

    delta_s = f(s) * a + wind

where f(s) is a smooth complex-valued field sampled once from a GP with RBF
kernel (approximated via Random Fourier Features), '*' is complex
multiplication, and 'wind' is a constant drift vector.

The agent starts at 0 and must reach a sparse goal neighborhood around (1+1i)
(i.e. the corner (1,1)).
"""

import gym
import numpy as np
import torch
from gym import spaces


class ComplexMaze(gym.Env):

    def __init__(
        self,
        dt: float = 0.05,
        num_steps: int = 125,
        n_features: int = 500,
        lengthscale: float = 1.0,
        goal_radius: float = 0.15,
        goal_reward: float = 1.0,
        wind_x: float = 0.0,
        wind_y: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.dt = float(dt)
        self.num_steps = int(num_steps)
        self.n_features = int(n_features)
        self.lengthscale = float(lengthscale)
        self.goal = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_radius = float(goal_radius)
        self.goal_reward = float(goal_reward)
        self.wind = np.array([float(wind_x), float(wind_y)], dtype=np.float32)

        # observation: (x, y, t)  in [0,1]^2 x [0,1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(2, ),
                                       dtype=np.float32)

        self._state = np.zeros(3, dtype=np.float32)
        self._step_count = 0

        # Sample the dynamics field (fixed for this instance)
        self._rng = np.random.RandomState(seed)
        self._sample_dynamics(self._rng)
        self._torch_cache_device = None

    # ------------------------------------------------------------------
    # Random Fourier Features for GP dynamics
    # ------------------------------------------------------------------
    def _sample_dynamics(self, rng: np.random.RandomState):
        """Sample a complex-valued dynamics field f: R^2 -> C via RFF."""
        D = self.n_features
        # RBF kernel frequencies: omega ~ N(0, 1/lengthscale^2 * I)
        self._omega = rng.randn(D, 2).astype(np.float32) / self.lengthscale
        self._bias = rng.uniform(0, 2 * np.pi, size=D).astype(np.float32)
        # Random weights: maps D cos features -> 2 outputs (real, imag)
        # Scale so the output has O(1) magnitude
        self._weights = rng.randn(D, 2).astype(np.float32) * np.sqrt(2.0 / D)

    def _dynamics_np(self, xy: np.ndarray) -> np.ndarray:
        """Evaluate f(s) for a single state. Returns (re, im)."""
        # xy: (2,)
        proj = self._omega @ xy + self._bias  # (D,)
        phi = np.cos(proj)  # (D,)
        out = phi @ self._weights  # (2,)  [re, im]
        return out

    def _ensure_torch_cache(self, device):
        if self._torch_cache_device != device:
            self._omega_t = torch.as_tensor(self._omega,
                                            dtype=torch.float32,
                                            device=device)
            self._bias_t = torch.as_tensor(self._bias,
                                           dtype=torch.float32,
                                           device=device)
            self._weights_t = torch.as_tensor(self._weights,
                                              dtype=torch.float32,
                                              device=device)
            self._torch_cache_device = device

    def _dynamics_batch_torch(self, xy: torch.Tensor) -> torch.Tensor:
        """Evaluate f(s) for a batch of states. Returns (N, 2) [re, im]."""
        self._ensure_torch_cache(xy.device)
        proj = xy @ self._omega_t.T + self._bias_t  # (N, D)
        phi = torch.cos(proj)  # (N, D)
        return phi @ self._weights_t  # (N, 2)

    # ------------------------------------------------------------------
    # Complex multiplication helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _complex_mul(z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Multiply two complex numbers given as (re, im) arrays."""
        return np.array(
            [z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self):
        self._state[...] = 0.0
        self._step_count = 0
        return self._state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        xy = self._state[:2].copy()
        f = self._dynamics_np(xy)
        delta = self._complex_mul(f, action) + self.wind
        next_xy = np.clip(xy + delta * self.dt, 0.0, 1.0)

        # Sparse reward
        dist_to_goal = float(np.linalg.norm(next_xy - self.goal))
        reward = self.goal_reward if dist_to_goal < self.goal_radius else 0.0

        self._state[:2] = next_xy
        self._step_count += 1
        done = self._step_count >= self.num_steps
        t_norm = float(self._step_count) / max(1, self.num_steps)
        self._state[2] = np.float32(np.clip(t_norm, 0.0, 1.0))

        return self._state.copy(), float(reward), bool(done), {}

    # ------------------------------------------------------------------
    # Visualization helpers (for callback)
    # ------------------------------------------------------------------
    def _grid(self, grid_res: int, device: torch.device, t: float = 0.0):
        xs = torch.linspace(0.0, 1.0, grid_res, device=device)
        ys = torch.linspace(0.0, 1.0, grid_res, device=device)
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

    def get_dynamics_grid(self, grid_res: int = 41, device=None):
        """Return the dynamics field f(s) on a grid, as (re, im) arrays."""
        device = torch.device("cpu") if device is None else device
        _, _, obs = self._grid(int(grid_res), device, t=0.0)
        f = self._dynamics_batch_torch(obs[:, :2])  # (N, 2)
        f = f.reshape(grid_res, grid_res, 2).detach().cpu().numpy()
        return f[:, :, 0], f[:, :, 1]
