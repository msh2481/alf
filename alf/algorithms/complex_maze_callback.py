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
"""Visualization callback for the ComplexMaze environment.

Produces per-iteration PNG files with:
  Row 0: dynamics field + replay trajectories + goal | Q(s,a) sweeps (agent 0)
  Row 1..N: Q-value heatmap + grad arrows | actor vector field
  (one row per agent copy, same layout as RotatorCallback)
"""

import os
from absl import logging
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import hsv_to_rgb
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import alf


@alf.configurable
class ComplexMazeCallback:

    def __init__(self,
                 debug_env=None,
                 log_every_n_steps: int = 100,
                 grid_res: int = 41,
                 grid_t: float = 0.5,
                 segment_scale: float = 0.05,
                 actor_segment_scale: float = 0.15,
                 quiver_width: float = 0.0022,
                 trajectory_alpha: float = 0.5,
                 trajectory_lw: float = 1.0,
                 max_traj_steps: int = 500,
                 name: str = "ComplexMazeCallback"):
        self._debug_env = debug_env
        self._log_every_n_steps = int(log_every_n_steps)
        self._max_traj_steps = int(max_traj_steps)
        self._grid_res = int(grid_res)
        self._grid_t = float(grid_t)
        self._arrow_scale = float(segment_scale)
        self._actor_arrow_scale = float(actor_segment_scale)
        self._quiver_width = float(quiver_width)
        self._trajectory_alpha = float(trajectory_alpha)
        self._trajectory_lw = float(trajectory_lw)
        self._name = name
        self._debug_count = 0
        greys = cm.get_cmap('bwr')
        self._cmap = ListedColormap(greys(np.linspace(0.25, 0.75, 256)))

    def _add_goal_patch(self,
                        ax,
                        facecolor="green",
                        alpha=0.2,
                        edgecolor="green",
                        linestyle="-"):
        """Add a goal circle clipped to the [0, 1]^2 plot region."""
        env = self._debug_env
        clip_rect = Rectangle((0, 0), 1.0, 1.0, transform=ax.transData)
        circle = Circle(tuple(env.goal),
                        env.goal_radius,
                        facecolor=facecolor,
                        alpha=alpha,
                        edgecolor=edgecolor,
                        linewidth=1.5,
                        linestyle=linestyle)
        circle.set_clip_path(clip_rect)
        ax.add_patch(circle)

    # ------------------------------------------------------------------
    # Replay buffer sampling
    # ------------------------------------------------------------------
    def _gather_trajectories(self, replay_buffer):
        """Gather the most recent ``max_traj_steps`` per env from the buffer.

        Returns:
            observations: (num_envs, T, obs_dim) numpy array
            step_types: (num_envs, T) numpy array
            or (None, None) if buffer is empty.
        """
        if replay_buffer is None or replay_buffer.total_size == 0:
            return None, None
        num_envs = replay_buffer._num_envs
        current_pos = replay_buffer._current_pos
        current_size = replay_buffer._current_size
        n = int(min(self._max_traj_steps, current_size.min().item()))
        if n <= 0:
            return None, None
        device = current_pos.device
        env_ids = torch.arange(num_envs, device=device).repeat_interleave(n)
        offsets = torch.arange(n, device=device)
        # Last n positions per env, in chronological order.
        positions = (current_pos[:, None] - n + offsets[None, :]).reshape(-1)
        obs = replay_buffer.get_field("observation", env_ids, positions)
        st = replay_buffer.get_field("step_type", env_ids, positions)
        observations = obs.reshape(num_envs, n,
                                   *obs.shape[1:]).detach().cpu().numpy()
        step_types = st.reshape(num_envs, n).detach().cpu().numpy()
        return observations, step_types

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
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
                f"{self._name}: expected 2D continuous action, got "
                f"{action_spec}")
            return

        observations, step_types = self._gather_trajectories(replay_buffer)
        if observations is None:
            return
        log_dir = os.environ.get("ALF_COMPLEX_MAZE_LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._create_and_save_plots(iter_number,
                                    observations,
                                    step_types,
                                    algorithms,
                                    num_copies,
                                    log_dir=log_dir)
        self._create_and_save_dynamics_plots(iter_number,
                                             algorithms,
                                             num_copies,
                                             log_dir=log_dir)

    # ------------------------------------------------------------------
    # Main figure
    # ------------------------------------------------------------------
    def _create_and_save_plots(self, iter_number, observations, step_types,
                               algorithms, num_copies, log_dir: str):
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

        # Row 0, left: dynamics + replay trajectories + goal
        self._plot_dynamics_and_trajectories(axes[0, 0], observations,
                                             step_types)

        # Row 0, right: Q(s,a) sweeps from agent 0
        first_im = None
        xs = torch.linspace(0.0, 1.0, self._grid_res, device=device)
        ys = torch.linspace(0.0, 1.0, self._grid_res, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        zz = torch.full_like(xx, self._grid_t)
        grid_obs = torch.stack([xx, yy, zz],
                               dim=-1).reshape(-1, 3).to(torch.float32)

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
                    return dist.sample((100, )).mean(0)

            if alg_idx == 0:
                self._plot_q_vs_action(axes[0, 1], alg, device=device)

            v_grid = self._debug_env.get_value_grid(q_batch_fn,
                                                    grid_res=self._grid_res,
                                                    device=device,
                                                    t=self._grid_t)
            zero_act = torch.zeros((grid_obs.shape[0], 2),
                                   dtype=torch.float32,
                                   device=device)
            g = grad_q_wrt_action_batch_fn(grid_obs, zero_act).reshape(
                self._grid_res, self._grid_res, 2).detach().cpu().numpy()
            gu, gv = g[:, :, 0], g[:, :, 1]
            gnorm = np.sqrt(gu * gu + gv * gv)
            gmax = float(np.max(gnorm)) if gnorm.size else 0.0
            if gmax > 0.0:
                gu = gu / (gmax + 1e-12)
                gv = gv / (gmax + 1e-12)
            im = self._plot_value_heatmap(axes[alg_idx + 1, 0],
                                          v_grid,
                                          gu * self._arrow_scale,
                                          gv * self._arrow_scale,
                                          alg_idx,
                                          t=self._grid_t)
            if first_im is None:
                first_im = im

            u_grid, v_vec_grid = self._debug_env.get_actor_grid(
                actor_batch_fn,
                grid_res=self._grid_res,
                device=device,
                t=self._grid_t)
            self._plot_actor_vector_field(axes[alg_idx + 1, 1],
                                          u_grid,
                                          v_vec_grid,
                                          alg_idx,
                                          t=self._grid_t)

        if first_im is not None:
            cax.set_visible(True)
            fig.colorbar(first_im, cax=cax, label="V(s)")
        plot_path = os.path.join(log_dir, f"{iter_number}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Written plot to {plot_path}")

    # ------------------------------------------------------------------
    # Dynamics field + trajectories (row 0, left)
    # ------------------------------------------------------------------
    def _plot_dynamics_and_trajectories(self, ax, observations, step_types):
        """Plot dynamics field and replay buffer trajectories.

        Args:
            observations: (num_envs, T, obs_dim) numpy array — full buffer
            step_types: (num_envs, T) numpy array (0=FIRST, 1=MID, 2=LAST)
        """
        env = self._debug_env
        # --- dynamics field as HSV image ---
        f_re, f_im = env.get_dynamics_grid(grid_res=self._grid_res)
        arg = np.arctan2(f_im, f_re)  # [-pi, pi]
        mag = np.sqrt(f_re**2 + f_im**2)
        mag_norm = mag / (mag.max() + 1e-8)

        hue = (arg + np.pi) / (2 * np.pi)  # [0, 1]
        sat = mag_norm
        val = np.ones_like(hue)
        hsv = np.stack([hue, sat, val], axis=-1)
        rgb = hsv_to_rgb(hsv)

        ax.imshow(rgb,
                  origin="lower",
                  extent=(0, 1, 0, 1),
                  aspect="equal",
                  alpha=0.6)

        # --- goal region (clipped to plot bounds) ---
        self._add_goal_patch(ax,
                             facecolor="green",
                             alpha=0.2,
                             edgecolor="green",
                             linestyle="-")

        # --- replay buffer trajectories as connected lines ---
        cmap = plt.get_cmap("tab10")
        num_envs = observations.shape[0]

        for e in range(num_envs):
            xy = observations[e, :, :2]  # (T, 2)
            st = step_types[e]  # (T,)
            color = cmap(e % cmap.N)

            # Split at episode boundaries (FIRST steps start new episodes)
            # Insert NaN at boundaries so matplotlib breaks the line
            breaks = np.where(st[1:] == 0)[0] + 1  # indices of FIRST steps
            if breaks.size > 0:
                xy_plot = xy.astype(np.float64)
                nan_row = np.full((1, 2), np.nan)
                segments = np.split(xy_plot, breaks)
                xy_plot = np.concatenate(
                    [s for seg in segments for s in (seg, nan_row)][:-1])
            else:
                xy_plot = xy
            ax.plot(xy_plot[:, 0],
                    xy_plot[:, 1],
                    color=color,
                    alpha=self._trajectory_alpha,
                    linewidth=self._trajectory_lw,
                    solid_capstyle="round")

        ax.set_title("Dynamics f(s) + replay trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        ax.text(0.02,
                0.97,
                "hue=arg(f)  sat=|f|",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ------------------------------------------------------------------
    # Q(s,a) sweeps (row 0, right)
    # ------------------------------------------------------------------
    def _plot_q_vs_action(self, ax, alg, device=None, sweep_points: int = 30):
        device = alf.get_default_device() if device is None else device
        sweep_points = int(sweep_points)
        a = torch.linspace(-1.0, 1.0, sweep_points,
                           device=device).to(torch.float32)

        states = torch.tensor([
            [0.5, 0.5, self._grid_t],
            [0.0, 0.0, self._grid_t],
            [1.0, 1.0, self._grid_t],
            [0.25, 0.75, self._grid_t],
            [0.75, 0.25, self._grid_t],
        ],
                              dtype=torch.float32,
                              device=device)
        state_labels = [
            "(0.5,0.5)", "(0,0)", "(1,1)", "(0.25,0.75)", "(0.75,0.25)"
        ]

        def q_batch_fn(obs_batch, act_batch):
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

        cmap_c = plt.get_cmap("tab10")
        state_handles = []

        for i, s in enumerate(states):
            obs = s.unsqueeze(0).expand(sweep_points, -1)
            color = cmap_c(i % cmap_c.N)

            act_x = torch.stack([a, torch.zeros_like(a)], dim=-1)
            qx = q_batch_fn(obs, act_x).detach().cpu().numpy()
            ax.plot(a.detach().cpu().numpy(),
                    qx,
                    color=color,
                    linestyle="-",
                    linewidth=1.8)

            act_y = torch.stack([torch.zeros_like(a), a], dim=-1)
            qy = q_batch_fn(obs, act_y).detach().cpu().numpy()
            ax.plot(a.detach().cpu().numpy(),
                    qy,
                    color=color,
                    linestyle="--",
                    linewidth=1.8)

            state_handles.append(
                Line2D([0], [0],
                       color=color,
                       linestyle="-",
                       linewidth=2.0,
                       label=state_labels[i]))

        ax.set_title("Agent 0: Q(s,a) sweeps")
        ax.set_xlabel("action component value")
        ax.set_ylabel("Q(s,a)")
        ax.grid(True, alpha=0.25)

        leg1 = ax.legend(handles=state_handles,
                         title="state s",
                         loc="upper left",
                         frameon=False)
        ax.add_artist(leg1)
        style_handles = [
            Line2D([0], [0],
                   color="black",
                   linestyle="-",
                   linewidth=2.0,
                   label="vary a_x (a_y=0)"),
            Line2D([0], [0],
                   color="black",
                   linestyle="--",
                   linewidth=2.0,
                   label="vary a_y (a_x=0)"),
        ]
        ax.legend(handles=style_handles, loc="lower left", frameon=False)

    # ------------------------------------------------------------------
    # Q-value heatmap (agent rows, left)
    # ------------------------------------------------------------------
    def _plot_value_heatmap(self, ax, v_grid, gu, gv, alg_idx, t):
        im = ax.imshow(v_grid,
                       origin="lower",
                       extent=(0, 1, 0, 1),
                       cmap=self._cmap,
                       aspect="equal")
        xs = np.linspace(0.0, 1.0, gu.shape[1])
        ys = np.linspace(0.0, 1.0, gu.shape[0])
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
        self._add_goal_patch(ax,
                             facecolor="none",
                             edgecolor="green",
                             linestyle="--")
        ax.set_title(f"Agent {alg_idx} Q(s,(0,0)) + grad (t={t:.3f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return im

    # ------------------------------------------------------------------
    # Actor vector field (agent rows, right)
    # ------------------------------------------------------------------
    def _plot_actor_vector_field(self, ax, u, v, alg_idx, t):
        xs = np.linspace(0.0, 1.0, u.shape[1])
        ys = np.linspace(0.0, 1.0, u.shape[0])
        xx, yy = np.meshgrid(xs, ys)
        u = u * self._actor_arrow_scale
        v = v * self._actor_arrow_scale
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
        self._add_goal_patch(ax,
                             facecolor="green",
                             alpha=0.15,
                             edgecolor="green",
                             linestyle="--")
        ax.set_title(f"Agent {alg_idx} actor (t={t:.3f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # Dynamics plots (separate file, like RotatorCallback)
    # ------------------------------------------------------------------
    def _create_and_save_dynamics_plots(self, iter_number, algorithms,
                                        num_copies, log_dir: str):
        agents_with_dyn = [(i, algorithms[i]) for i in range(num_copies)
                           if hasattr(algorithms[i], 'predict_next')]
        if not agents_with_dyn:
            return

        device = alf.get_default_device()

        xs = torch.linspace(0.0, 1.0, self._grid_res, device=device)
        ys = torch.linspace(0.0, 1.0, self._grid_res, device=device)
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")
        zz = torch.full_like(xx, self._grid_t)
        grid_obs = torch.stack([xx, yy, zz],
                               dim=-1).reshape(-1, 3).to(torch.float32)
        n_grid = grid_obs.shape[0]

        xs_np = np.linspace(0.0, 1.0, self._grid_res)
        ys_np = np.linspace(0.0, 1.0, self._grid_res)
        xx_np, yy_np = np.meshgrid(xs_np, ys_np)

        _ACTIONS = [
            (torch.tensor([1.0, 0.0]), "a=(+1, 0)"),
            (torch.tensor([-1.0, 0.0]), "a=(-1, 0)"),
            (torch.tensor([0.0, 1.0]), "a=(0, +1)"),
            (torch.tensor([0.0, -1.0]), "a=(0, -1)"),
        ]
        n_actions = len(_ACTIONS)
        n_rows = len(agents_with_dyn)

        fig, axes = plt.subplots(n_rows,
                                 n_actions,
                                 figsize=(6 * n_actions, 6 * n_rows),
                                 constrained_layout=True)
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, (alg_idx, alg) in enumerate(agents_with_dyn):
            for col, (a_vec, label) in enumerate(_ACTIONS):
                ax = axes[row, col]
                action = a_vec.to(device).unsqueeze(0).expand(n_grid, -1)

                with torch.no_grad():
                    s_next, r_hat = alg.predict_next(grid_obs, action)

                disp = (s_next[:, :2] - grid_obs[:, :2]).detach().cpu().numpy()
                disp = disp.reshape(self._grid_res, self._grid_res, 2)
                du, dv = disp[:, :, 0], disp[:, :, 1]

                r_grid = r_hat.detach().cpu().numpy().reshape(
                    self._grid_res, self._grid_res)
                im = ax.imshow(r_grid,
                               origin="lower",
                               extent=(0, 1, 0, 1),
                               cmap="RdYlGn",
                               aspect="equal",
                               alpha=0.3)
                fig.colorbar(im, ax=ax, shrink=0.8, label="r_hat")

                mag = np.sqrt(du**2 + dv**2)
                max_mag = float(mag.max()) if mag.size else 0.0
                if max_mag > 1e-8:
                    du = du / max_mag * self._arrow_scale
                    dv = dv / max_mag * self._arrow_scale

                ax.quiver(xx_np,
                          yy_np,
                          du,
                          dv,
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
                ax.set_title(f"Agent {alg_idx} dynamics {label} "
                             f"(t={self._grid_t:.2f})")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_aspect("equal")

        plot_path = os.path.join(log_dir, f"{iter_number}_dynamics.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        logging.info(f"Written dynamics plot to {plot_path}")
