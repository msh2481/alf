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

import os
from absl import logging
import numpy as np
import torch
import torch.distributions as td
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

import alf


@alf.configurable
class RotatorCallback:

    def __init__(self,
                 debug_env=None,
                 log_every_n_steps: int = 100,
                 num_samples: int = 4000,
                 grid_res: int = 41,
                 vmin: float = -0.5,
                 vmax: float = 5.0,
                 segment_scale: float = 0.05,
                 quiver_width: float = 0.0022,
                 name: str = "RotatorCallback"):
        self._debug_env = debug_env
        self._log_every_n_steps = int(log_every_n_steps)
        self._num_samples = int(num_samples)
        self._grid_res = int(grid_res)
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._arrow_scale = float(segment_scale)
        self._quiver_width = float(quiver_width)
        self._name = name
        self._debug_count = 0
        # Use first half of Greys colormap (light colors) for black quiver visibility
        greys = cm.get_cmap('Greys')
        self._cmap = ListedColormap(greys(np.linspace(0, 0.5, 256)))

    def _sample_from_replay_buffer(self, replay_buffer):
        if replay_buffer is None or replay_buffer.total_size == 0:
            return None, None, None
        n = min(self._num_samples, replay_buffer.total_size.item())
        batch_info = replay_buffer._sample(batch_size=n, batch_length=1)
        observations = replay_buffer.get_field("observation",
                                               batch_info.env_ids,
                                               batch_info.positions)
        actions = replay_buffer.get_field("action", batch_info.env_ids,
                                          batch_info.positions)
        return observations, actions, batch_info.env_ids

    def __call__(self,
                 replay_buffer,
                 algorithms,
                 action_spec,
                 num_copies,
                 iter_number=None):
        if iter_number is None:
            iter_number = self._debug_count
        self._debug_count += 1
        if iter_number % self._log_every_n_steps != 0:
            return

        assert self._debug_env is not None, "debug_env must be provided"
        if action_spec.is_discrete or tuple(action_spec.shape) != (2, ):
            logging.warning(
                f"{self._name}: expected 2D continuous action, got {action_spec}"
            )
            return

        observations, actions, env_ids = self._sample_from_replay_buffer(
            replay_buffer)
        if observations is None:
            return

        os.makedirs("logs", exist_ok=True)
        self._create_and_save_plots(iter_number, observations, actions,
                                    env_ids, algorithms, num_copies)

    def _create_and_save_plots(self, iter_number, observations, actions,
                               env_ids, algorithms, num_copies):
        device = alf.get_default_device()
        nrows = num_copies + 1
        fig = plt.figure(figsize=(16, 6 * nrows), constrained_layout=True)
        gs = fig.add_gridspec(nrows=nrows, ncols=3, width_ratios=[1, 0.05, 1])
        axes = np.empty((nrows, 2), dtype=object)
        for r in range(nrows):
            for c in range(2):
                axes[r, c] = fig.add_subplot(gs[r, [0, 2][c]])
        cax = fig.add_subplot(gs[1:, 1])
        cax.set_visible(False)

        self._plot_replay_buffer_vector_field(axes[0, 0], observations,
                                              actions, env_ids)
        axes[0, 1].axis("off")

        first_im = None
        xs = torch.linspace(-1.0, 1.0, self._grid_res, device=device)
        ys = torch.linspace(-1.0, 1.0, self._grid_res, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        grid_obs = torch.stack([xx, yy], dim=-1).reshape(-1,
                                                         2).to(torch.float32)
        for alg_idx in range(num_copies):
            alg = algorithms[alg_idx]

            def q_batch_fn(obs_batch: torch.Tensor, act_batch: torch.Tensor):
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                with torch.no_grad():
                    q, _ = alg._compute_critics(alg._critic_networks,
                                                obs_batch,
                                                act_batch,
                                                critics_state=(),
                                                replica_min=True,
                                                apply_reward_weights=True)
                return torch.as_tensor(q).reshape(-1)

            def grad_q_wrt_action_batch_fn(obs_batch: torch.Tensor,
                                           act_batch: torch.Tensor):
                obs_batch = obs_batch.to(device).detach()
                act_batch = act_batch.to(device).detach().requires_grad_(True)
                with torch.enable_grad():
                    q, _ = alg._compute_critics(alg._critic_networks,
                                                obs_batch,
                                                act_batch,
                                                critics_state=(),
                                                replica_min=True,
                                                apply_reward_weights=True)
                    q = torch.as_tensor(q).reshape(-1)
                    (g, ) = torch.autograd.grad(q.sum(), act_batch)
                return g

            def actor_batch_fn(obs_batch: torch.Tensor):
                obs_batch = obs_batch.to(device)
                with torch.no_grad():
                    dist, _ = alg._actor_network(obs_batch, state=())
                base = dist.base_dist if isinstance(
                    dist, td.TransformedDistribution) else dist
                if isinstance(base, td.Independent):
                    base = base.base_dist
                return base.mean

            v_grid = self._debug_env.get_value_grid(q_batch_fn,
                                                    grid_res=self._grid_res,
                                                    device=device)
            zero_act = torch.zeros_like(grid_obs)
            g = grad_q_wrt_action_batch_fn(grid_obs, zero_act).reshape(
                self._grid_res, self._grid_res, 2).detach().cpu().numpy()
            gu = g[:, :, 0]
            gv = g[:, :, 1]
            gnorm = np.sqrt(gu * gu + gv * gv)
            gmax = float(np.max(gnorm)) if gnorm.size else 0.0
            if gmax > 0.0:
                gu = gu / (gmax + 1e-12)
                gv = gv / (gmax + 1e-12)
            im = self._plot_value_heatmap(axes[alg_idx + 1, 0], v_grid,
                                          gu * self._arrow_scale,
                                          gv * self._arrow_scale, alg_idx)
            if first_im is None:
                first_im = im

            u_grid, v_vec_grid = self._debug_env.get_actor_grid(
                actor_batch_fn, grid_res=self._grid_res, device=device)
            self._plot_actor_vector_field(axes[alg_idx + 1, 1], u_grid,
                                          v_vec_grid, alg_idx)

        if first_im is not None:
            cax.set_visible(True)
            fig.colorbar(first_im, cax=cax, label="V(s)")
        plot_path = f"logs/{iter_number}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Written plot to {plot_path}")

    def _plot_value_heatmap(self, ax, v_grid: np.ndarray, gu: np.ndarray,
                            gv: np.ndarray, alg_idx: int):
        im = ax.imshow(v_grid,
                       origin="lower",
                       extent=(-1, 1, -1, 1),
                       cmap=self._cmap,
                       vmin=self._vmin,
                       vmax=self._vmax,
                       aspect="equal")
        xs = np.linspace(-1.0, 1.0, gu.shape[1])
        ys = np.linspace(-1.0, 1.0, gu.shape[0])
        xx, yy = np.meshgrid(xs, ys)
        ax.quiver(xx,
                  yy,
                  gu,
                  gv,
                  angles="xy",
                  scale_units="xy",
                  scale=1.0,
                  width=self._quiver_width,
                  headwidth=2.0,
                  headlength=3.0,
                  headaxislength=2.5,
                  pivot="mid",
                  color="black",
                  alpha=0.9)
        ax.set_title(f"Agent {alg_idx} Q(s,(0,0)) and its gradient")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return im

    def _plot_actor_vector_field(self, ax, u: np.ndarray, v: np.ndarray,
                                 alg_idx: int):
        xs = np.linspace(-1.0, 1.0, u.shape[1])
        ys = np.linspace(-1.0, 1.0, u.shape[0])
        xx, yy = np.meshgrid(xs, ys)
        u = u * self._arrow_scale
        v = v * self._arrow_scale
        ax.quiver(xx,
                  yy,
                  u,
                  v,
                  angles="xy",
                  scale_units="xy",
                  scale=1.0,
                  width=self._quiver_width,
                  headwidth=2.0,
                  headlength=3.0,
                  headaxislength=2.5,
                  pivot="mid",
                  alpha=0.9)
        ax.set_title(f"Agent {alg_idx} actor")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")

    def _plot_replay_buffer_vector_field(self, ax, observations, actions,
                                         env_ids):
        obs = torch.as_tensor(observations).detach().cpu().numpy().reshape(
            -1, 2)
        act = torch.as_tensor(actions).detach().cpu().numpy().reshape(-1, 2)
        env_ids = torch.as_tensor(env_ids).detach().cpu().numpy().reshape(-1)

        uniq = np.unique(env_ids)
        cmap = plt.get_cmap("tab10")
        color_of = {int(e): cmap(i % cmap.N) for i, e in enumerate(uniq)}
        colors = np.array([color_of[int(e)] for e in env_ids],
                          dtype=np.float32)
        u = act[:, 0] * self._arrow_scale
        v = act[:, 1] * self._arrow_scale
        ax.quiver(obs[:, 0],
                  obs[:, 1],
                  u,
                  v,
                  angles="xy",
                  scale_units="xy",
                  scale=1.0,
                  width=self._quiver_width,
                  headwidth=2.0,
                  headlength=3.0,
                  headaxislength=2.5,
                  pivot="mid",
                  color=colors,
                  alpha=0.75)

        ax.set_title("Replay buffer")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
