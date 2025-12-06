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

import os
import torch
import torch.distributions as td
import numpy as np
from absl import logging
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import alf
from alf.tensor_specs import TensorSpec


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    return mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))


@alf.configurable
class DebugCallback:
    """Callback for debugging concurrent algorithms with visualization and metrics."""

    def __init__(self,
                 debug_env=None,
                 log_every_n_steps: int = 100,
                 num_samples: int = 1000,
                 name: str = "DebugCallback"):
        self._debug_env = debug_env
        self._log_every_n_steps = log_every_n_steps
        self._num_samples = num_samples
        self._debug_count = 0
        self._name = name

    def _sample_from_replay_buffer(self, replay_buffer):
        """Sample observations, actions, and rewards from replay buffer."""
        if replay_buffer is None or replay_buffer.total_size == 0:
            return None, None, None
        num_samples = min(self._num_samples, replay_buffer.total_size.item())
        batch_info = replay_buffer._sample(batch_size=num_samples,
                                           batch_length=1)
        observations = replay_buffer.get_field('observation',
                                               batch_info.env_ids,
                                               batch_info.positions)
        actions = replay_buffer.get_field('action', batch_info.env_ids,
                                          batch_info.positions)
        rewards = replay_buffer.get_field('reward', batch_info.env_ids,
                                          batch_info.positions)
        return observations, actions, rewards

    def _create_get_q_values_fn(self, algorithms, action_spec, device):
        """Create a function to get Q-values for a given algorithm index.
        
        Returns a function that takes (alg_index, obs, act) and returns q_values tensor.
        Handles device conversion and batch dimension automatically.
        """

        def get_q_values_fn(alg_index, obs, act):
            obs = obs.to(device)
            act = act.to(device)
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            alg = algorithms[alg_index]
            if action_spec.is_discrete:
                q_values, _ = alg._compute_critics(alg._critic_networks,
                                                   obs,
                                                   None,
                                                   critics_state=(),
                                                   replica_min=True,
                                                   apply_reward_weights=True)
                q_values = q_values.gather(1, act.unsqueeze(1)).squeeze(1)
            else:
                q_values, _ = alg._compute_critics(alg._critic_networks,
                                                   obs,
                                                   act,
                                                   critics_state=(),
                                                   replica_min=True,
                                                   apply_reward_weights=True)
            return q_values[0].item()

        return get_q_values_fn

    def _create_get_actor_fn(self, algorithms, action_spec, device):
        """Create a function to get actor distribution parameters.

        Returns a function that takes (alg_index, obs) and returns a dict
        with 'mean' and 'std' scalars extracted from the actor's output distribution.
        Handles device conversion and batch dimension automatically.

        Args:
            algorithms: List of algorithm instances
            action_spec: Action tensor spec
            device: Device to run computation on

        Returns:
            Callable: Function (alg_index, obs) -> {'mean': float, 'std': float}
        """

        def get_actor_fn(alg_index, obs):
            obs = obs.to(device)
            obs = obs.unsqueeze(0)

            alg = algorithms[alg_index]

            # Check if algorithm has actor network
            if not hasattr(alg,
                           '_actor_network') or alg._actor_network is None:
                raise ValueError(
                    f"Algorithm {alg_index} does not have an actor network. "
                    "Actor visualization requires continuous action space.")

            # Call actor network to get distribution
            action_dist, _ = alg._actor_network(obs, state=())

            # Handle different distribution types
            if isinstance(action_dist, td.TransformedDistribution):
                base_dist = action_dist.base_dist
            else:
                base_dist = action_dist

            # Handle Independent/DiagMultivariateNormal wrapping
            if isinstance(base_dist, td.Independent):
                base_dist = base_dist.base_dist

            # Extract mean and std from the base Normal distribution
            # base_dist should now be td.Normal
            mean = base_dist.loc[0]
            std = base_dist.scale[0]

            # For BipolarChain, action is 1D, so take single value
            if mean.numel() > 1:
                # Multi-dimensional continuous action - take first dimension
                mean = mean[0].item()
                std = std[0].item()
            else:
                mean = mean.item()
                std = std.item()

            # Clamp std to prevent numerical issues
            std = max(std, 1e-8)

            return {'mean': mean, 'std': std}

        return get_actor_fn

    def __call__(self,
                 replay_buffer,
                 algorithms,
                 action_spec,
                 num_copies,
                 iter_number=None):
        """Call the debug callback with data from the algorithm.

        Args:
            replay_buffer: replay buffer to sample from
            algorithms: list of algorithm instances
            action_spec: action tensor spec
            num_copies: number of algorithm copies
            iter_number: iteration number (optional, defaults to debug_count)
        """
        if iter_number is None:
            iter_number = self._debug_count
        self._debug_count += 1
        if iter_number % self._log_every_n_steps != 0:
            return

        observations, actions, rewards = self._sample_from_replay_buffer(
            replay_buffer)
        if observations is None:
            return

        assert self._debug_env is not None, "debug_env is None - must be passed to DebugCallback"
        assert hasattr(self._debug_env, 'get_q_value_table'), \
            f"Environment {type(self._debug_env)} missing get_q_value_table method"
        assert hasattr(self._debug_env, 'get_transition_counts_table'), \
            f"Environment {type(self._debug_env)} missing get_transition_counts_table method"

        device = alf.get_default_device()
        get_q_values_fn = self._create_get_q_values_fn(algorithms, action_spec,
                                                       device)

        # Create actor function for continuous action spaces
        get_actor_fn = None
        if not action_spec.is_discrete:
            try:
                get_actor_fn = self._create_get_actor_fn(
                    algorithms, action_spec, device)
            except Exception as e:
                logging.warning(f"Failed to create actor function: {e}")
                logging.warning("Actor visualization will be skipped.")

        os.makedirs('logs', exist_ok=True)
        log_file_path = f'logs/{iter_number}.txt'

        with open(log_file_path, 'w') as f:
            self._write_basic_stats(f, observations, rewards, replay_buffer)
            f.write("=" * 40 + "\n")

        self._create_and_save_plots(iter_number, replay_buffer, num_copies,
                                    get_q_values_fn, get_actor_fn)
        logging.info(f"Written debug metrics to {log_file_path}")

    def _write_basic_stats(self, f, observations, rewards, replay_buffer):
        """Write basic statistics to file."""
        num_samples = observations.shape[0]
        total_size = replay_buffer.total_size.item(
        ) if replay_buffer is not None else 0
        f.write(f"Sampled {num_samples} out of {total_size} experiences\n")
        if rewards is not None:
            mean_reward = rewards.mean().item()
            f.write(f"\nMean reward: {mean_reward:.4f}\n")

    def _create_and_save_plots(self,
                               iter_number,
                               replay_buffer,
                               num_copies,
                               get_q_values_fn,
                               get_actor_fn=None):
        """Create and save visualization plots."""
        k = self._debug_env.k
        positions = list(range(-k, k + 1))

        fig, axes = plt.subplots(num_copies + 1,
                                 3,
                                 figsize=(36, 6 * (num_copies + 1)))

        transition_counts = self._debug_env.get_transition_counts_table(
            replay_buffer)
        self._plot_transition_counts(axes[0, :2], transition_counts, k,
                                     positions)

        for i in range(num_copies):

            def q_func(obs, action):
                return get_q_values_fn(i, obs, action)

            self._plot_q_values(axes[i + 1, :2], i, k, positions, q_func)

            # Actor probabilities plotting (third column)
            if get_actor_fn is not None:

                def actor_func(obs):
                    return get_actor_fn(i, obs)

                self._plot_actor_probabilities(axes[i + 1, 2], i, k, positions,
                                               actor_func)
            else:
                # Discrete actions - add explanatory text
                axes[i + 1, 2].text(
                    0.5,
                    0.5,
                    'Actor visualization\nonly for continuous\naction spaces',
                    ha='center',
                    va='center',
                    transform=axes[i + 1, 2].transAxes,
                    fontsize=12)
                axes[i + 1, 2].axis('off')

        plt.tight_layout()
        plot_path = f'logs/{iter_number}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Written plot to {plot_path}")

    def _plot_transition_counts(self, axes_row, transition_counts, k,
                                positions):
        """Plot transition counts for each action."""
        for action_idx, action_name in enumerate(['Left', 'Right']):
            ax = axes_row[action_idx]
            data = transition_counts[:, :, action_idx].T
            im = ax.imshow(data,
                           aspect='auto',
                           cmap=truncate_colormap("Blues", 0, 0.5),
                           origin='lower',
                           vmin=0,
                           vmax=5)
            ax.set_xlabel('Position')
            ax.set_ylabel('Time')
            ax.set_title(f'Transitions {action_name}')
            ax.set_xticks(range(0, 2 * k + 1, max(1, (2 * k + 1) // 8)))
            ax.set_xticklabels([
                positions[j]
                for j in range(0, 2 * k + 1, max(1, (2 * k + 1) // 8))
            ])
            ax.set_yticks(range(k + 1))
            plt.colorbar(im, ax=ax)

            for pos_idx in range(2 * k + 1):
                for time_idx in range(k + 1):
                    value = data[time_idx, pos_idx]
                    ax.text(pos_idx,
                            time_idx,
                            f"{int(value) if not np.isnan(value) else ''}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=10)

    def _plot_q_values(self, axes_row, alg_index, k, positions,
                       get_q_values_fn):
        """Plot Q-values for a given algorithm."""
        q_values = self._debug_env.get_q_value_table(get_q_values_fn)

        for action_idx, action_name in enumerate(['Left', 'Right']):
            ax = axes_row[action_idx]
            data = q_values[:, :, action_idx].T
            im = ax.imshow(data,
                           aspect='auto',
                           cmap='bwr',
                           origin='lower',
                           vmin=-0.5,
                           vmax=0.5)
            ax.set_xlabel('Position')
            ax.set_ylabel('Time')
            ax.set_title(f'Algorithm {alg_index} Q-values {action_name}')
            ax.set_xticks(range(0, 2 * k + 1, max(1, (2 * k + 1) // 8)))
            ax.set_xticklabels([
                positions[j]
                for j in range(0, 2 * k + 1, max(1, (2 * k + 1) // 8))
            ])
            ax.set_yticks(range(k + 1))
            plt.colorbar(im, ax=ax)

            for pos_idx in range(2 * k + 1):
                for time_idx in range(k + 1):
                    value = data[time_idx, pos_idx]
                    if np.isnan(value):
                        continue
                    ax.text(pos_idx,
                            time_idx,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=10)

    def _plot_actor_probabilities(self, ax, alg_index, k, positions,
                                  actor_callable):
        """Plot actor output probabilities for a given algorithm.

        Creates a heatmap showing P(action >= 0) for each state, with annotations
        displaying the mean and standard deviation of the actor's output distribution.

        Args:
            ax: Matplotlib axis to plot on
            alg_index: Index of the algorithm
            k: Environment parameter (max position/time)
            positions: List of position values [-k, ..., k]
            actor_callable: Function (obs) -> {'mean': float, 'std': float}
        """
        actor_probs = self._debug_env.get_actor_table(actor_callable)

        # Transpose for plotting (time on y-axis, position on x-axis)
        data = actor_probs.T

        # Create heatmap with probability colormap
        im = ax.imshow(data,
                       aspect='auto',
                       cmap='RdYlGn',
                       origin='lower',
                       vmin=0.0,
                       vmax=1.0)

        # Set labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
        ax.set_title(f'Algorithm {alg_index} Actor P(Right)')

        # Set ticks (same pattern as Q-value plots)
        ax.set_xticks(range(0, 2 * k + 1, max(1, (2 * k + 1) // 8)))
        ax.set_xticklabels([
            positions[j] for j in range(0, 2 * k + 1, max(1, (2 * k + 1) // 8))
        ])
        ax.set_yticks(range(k + 1))

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Annotate cells with mean and std
        for pos_idx in range(2 * k + 1):
            for time_idx in range(k + 1):
                prob = data[time_idx, pos_idx]
                if np.isnan(prob):
                    continue

                # Get mean and std for annotation
                position = positions[pos_idx]
                obs = self._debug_env.state_to_observation(position, time_idx)
                obs_tensor = torch.from_numpy(obs)
                actor_output = actor_callable(obs_tensor)
                mean, std = actor_output['mean'], actor_output['std']

                # Create annotation text
                ax.text(pos_idx,
                        time_idx,
                        f"μ={mean:.2f}\nσ={std:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8)
