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
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import torch
import torch.nn as nn
from absl import logging
import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils.common import slice_nested, scatter_and_sum_nested
from alf.nest_formatter import format_nest
from alf.utils.dist_utils import distributions_to_params, params_to_distributions, extract_spec


@alf.configurable
class ConcurrentAlgorithm(OffPolicyAlgorithm):

    def __init__(
        self,
        observation_spec,
        action_spec,
        algorithm_ctor: Callable,
        num_copies: int = 2,
        reward_spec=TensorSpec(()),
        env=None,
        config: Optional[TrainerConfig] = None,
        checkpoint: Optional[str] = None,
        optimizer=None,
        debug_summaries: bool = False,
        name: str = "ConcurrentAlgorithm",
        batch_size=None,
        env_counts=None,
        unroll_length=None,
        mini_batch_length=None,
        use_exploration_seeds: bool = True,
        use_parallel_training: bool = True,
        num_parallel_workers: Optional[int] = None,
        video_record_interval: int | None = None,
        return_logging_interval: int = 100,
        agent_reset_period: int | None = None,
        debug_callback=None,
    ):
        assert batch_size is not None, "batch_size must be provided"
        assert env_counts is not None, "env_counts must be provided"
        assert unroll_length is not None, "unroll_length must be provided"
        assert mini_batch_length is not None, "mini_batch_length must be provided"
        assert batch_size % num_copies == 0, f"batch_size {batch_size} must be a multiple of num_copies {num_copies}"
        assert env_counts % num_copies == 0, f"env_counts {env_counts} must be a multiple of num_copies {num_copies}"
        self._batch_size = batch_size
        self._env_counts = env_counts
        self._unroll_length = unroll_length
        self._mini_batch_length = mini_batch_length

        temp_alg = algorithm_ctor(observation_spec=observation_spec,
                                  action_spec=action_spec,
                                  reward_spec=reward_spec)
        is_on_policy = temp_alg.on_policy
        train_state_spec = [
            temp_alg.train_state_spec for _ in range(num_copies)
        ]
        rollout_state_spec = [
            temp_alg.rollout_state_spec for _ in range(num_copies)
        ]
        predict_state_spec = [
            temp_alg.predict_state_spec for _ in range(num_copies)
        ]
        del temp_alg

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            is_on_policy=is_on_policy,
            env=env,
            config=config,
            checkpoint=checkpoint,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name,
        )

        self._num_copies = num_copies
        self._algorithm_ctor = algorithm_ctor
        self._use_exploration_seeds = use_exploration_seeds
        self._algorithm_base_kwargs = {
            "observation_spec": observation_spec,
            "action_spec": action_spec,
            "reward_spec": reward_spec,
            "env": None,
            "config": config,
            "debug_summaries": debug_summaries,
        }
        self._algorithm_name = name

        def make_algorithm(i):
            kwargs = {
                **self._algorithm_base_kwargs,
                "name": f"{name}_copy_{i}",
            }
            if use_exploration_seeds:
                kwargs["exploration_seed"] = i
            return algorithm_ctor(**kwargs)

        self._algorithms = nn.ModuleList(
            [make_algorithm(i) for i in range(num_copies)])

        # Initialize parallel training executor
        self._use_parallel_training = use_parallel_training
        if self._use_parallel_training:
            self._num_workers = num_parallel_workers or num_copies
            self._executor = ThreadPoolExecutor(max_workers=self._num_workers)
        else:
            self._executor = None

        # Video recording
        self._video_record_interval = video_record_interval
        self._return_logging_interval = return_logging_interval
        self._train_step_counter = 0

        # Per-algorithm episode return tracking
        self._per_env_cumulative_reward = torch.zeros(env_counts)
        self._per_alg_returns: dict[int, list[tuple[int, float]]] = {
            i: []
            for i in range(num_copies)
        }
        self._total_env_steps = 0

        # Per-algorithm loss tracking
        self._per_alg_actor_losses: dict[int, list[tuple[int, float]]] = {
            i: []
            for i in range(num_copies)
        }
        self._per_alg_critic_losses: dict[int, list[tuple[int, float]]] = {
            i: []
            for i in range(num_copies)
        }

        self._agent_reset_period = agent_reset_period
        self._next_agent_to_reset = 0
        self._debug_callback = debug_callback

    def _get_most_recently_reset_agent(self) -> int:
        """Get the index of the most recently reset agent."""
        return (self._next_agent_to_reset + self._num_copies -
                1) % self._num_copies

    def _reset_agent(self, agent_idx: int):
        """Reinitialize agent at given index with fresh parameters."""
        logging.warning(f"Resetting agent {agent_idx} parameters")
        kwargs = {
            **self._algorithm_base_kwargs,
            "name":
                f"{self._algorithm_name}_copy_{agent_idx}",
        }
        if self._use_exploration_seeds:
            kwargs["exploration_seed"] = agent_idx
        new_alg = self._algorithm_ctor(**kwargs)
        old_alg = self._algorithms[agent_idx]
        for old_param, new_param in zip(old_alg.parameters(),
                                        new_alg.parameters()):
            old_param.data.copy_(new_param.data)
        logging.info(f"Reset agent {agent_idx} parameters")

    def close(self):
        """Clean up resources, including shutting down the thread pool executor."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        """Destructor to ensure executor is cleaned up."""
        self.close()

    def get_initial_predict_state(self, batch_size):
        return [
            alg.get_initial_predict_state(batch_size)
            for alg in self._algorithms
        ]

    def get_initial_rollout_state(self, batch_size):
        return [
            alg.get_initial_rollout_state(batch_size)
            for alg in self._algorithms
        ]

    def get_initial_train_state(self, batch_size):
        return [
            alg.get_initial_train_state(batch_size) for alg in self._algorithms
        ]

    def _slice_batch(self, *args, time_major=False):
        """Route each batch element to the appropriate algorithm copy.

        Args:
            *args: Arbitrary number of arguments to route. First arg is used to determine batch size.

        Returns:
            dict: mapping algorithm index -> (sliced_arg1, sliced_arg2, ..., batch_indices)
        """
        n = alf.nest.get_nest_size(args[0], dim=1 if time_major else 0)
        assert n % self._num_copies == 0, f"n {n} must be a multiple of num_copies {self._num_copies}"
        device = next(iter(alf.nest.flatten(args[0]))).device
        sliced = {}
        for i in range(self._num_copies):
            indices = torch.arange(i, n, self._num_copies, device=device)
            sliced_args = []
            for arg in args:
                if isinstance(arg, list) and len(arg) == self._num_copies:
                    # This case is for state, which we store as a list
                    sliced_args.append(arg[i])
                else:
                    sliced_args.append(
                        slice_nested(arg, indices, time_major=time_major))
            sliced[i] = (*sliced_args, indices)
        return sliced

    def rollout_step(self, inputs: TimeStep, state) -> AlgStep:
        assert alf.nest.get_nest_size(
            inputs, dim=0
        ) == self._env_counts, f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

        # Track per-algorithm episode returns
        self._track_episode_returns(inputs)

        sliced = self._slice_batch(inputs, state)

        results = {}
        new_states = [None] * self._num_copies
        for alg_idx, (
                sliced_time_step,
                sliced_state,
                batch_indices,
        ) in sliced.items():
            alg_step = self._algorithms[alg_idx].rollout_step(
                sliced_time_step, sliced_state)
            new_states[alg_idx] = alg_step.state
            results[alg_idx] = (alg_step._replace(state=()), batch_indices)
        return scatter_and_sum_nested(
            results, self._env_counts)._replace(state=new_states)

    def _track_episode_returns(self, inputs: TimeStep):
        device = inputs.reward.device
        if self._per_env_cumulative_reward.device != device:
            self._per_env_cumulative_reward = self._per_env_cumulative_reward.to(
                device)
        self._total_env_steps += self._env_counts
        self._per_env_cumulative_reward += inputs.reward
        is_last = inputs.is_last()
        if is_last.any():
            for env_idx in is_last.nonzero(as_tuple=True)[0]:
                alg_idx = env_idx.item() % self._num_copies
                episode_return = self._per_env_cumulative_reward[env_idx].item(
                )
                num_episodes = len(self._per_alg_returns[alg_idx]) + 1
                logging.info(
                    f"Agent {alg_idx} episode {num_episodes} done: "
                    f"return={episode_return:.2f}, env_steps={self._total_env_steps}"
                )
                self._per_alg_returns[alg_idx].append(
                    (self._total_env_steps, episode_return))
                self._per_env_cumulative_reward[env_idx] = 0.0

    def get_per_algorithm_returns(self) -> dict[int, list[tuple[int, float]]]:
        """Get recorded episode returns for each sub-algorithm.
        
        Returns:
            Dict mapping algorithm index to list of (env_steps, episode_return) tuples.
        """
        return self._per_alg_returns

    def _get_indices_of_unique(self, observations, actions):

        def unique(x, dim):
            unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
            perm = torch.arange(inverse.size(dim),
                                dtype=inverse.dtype,
                                device=inverse.device)
            inverse, perm = inverse.flip([dim]), perm.flip([dim])
            return unique, inverse.new_empty(unique.size(dim)).scatter_(
                dim, inverse, perm)

        batch_size, obs_dim = observations.shape
        assert actions.shape == (
            batch_size, ), f"actions shape: {actions.shape}"
        obs_action_pairs = torch.cat(
            [observations, actions.unsqueeze(1)], dim=1)
        _, indices_of_unique = unique(obs_action_pairs, dim=0)
        # now sample len(observations) indices with replacement from indices
        n_full = len(observations)
        n_unique = len(indices_of_unique)
        indices = torch.randint(0,
                                n_unique, (n_full, ),
                                device=indices_of_unique.device)
        return indices_of_unique[indices]

    def _take_from_indices(self, data, indices):
        spec = extract_spec(data)
        data_params = distributions_to_params(data)
        masked = alf.nest.map_structure(lambda x: x[indices], data_params)
        return params_to_distributions(masked, spec)

    def train_step(self, inputs: TimeStep, state, rollout_info) -> AlgStep:
        total_batch_size = self._mini_batch_length * self._batch_size
        assert alf.nest.get_nest_size(
            inputs, dim=0
        ) == total_batch_size, f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

        active_agent_idx = self._get_most_recently_reset_agent()
        active_state = state[active_agent_idx] if isinstance(state,
                                                             list) else state
        active_rollout_info = rollout_info[active_agent_idx] if isinstance(
            rollout_info, list) else rollout_info

        alg_step = self._algorithms[active_agent_idx].train_step(
            inputs, active_state, active_rollout_info)

        if isinstance(state, list):
            new_states = list(state)
            new_states[active_agent_idx] = alg_step.state
        else:
            new_states = [alg_step.state] + [None] * (self._num_copies - 1)

        return alg_step._replace(state=new_states)

    def calc_loss(self, info) -> LossInfo:
        assert alf.nest.get_nest_shape(info)[:2] == (
            self._mini_batch_length,
            self._batch_size), f"info shape: {alf.nest.get_nest_shape(info)}"

        active_agent_idx = self._get_most_recently_reset_agent()
        loss_info = self._algorithms[active_agent_idx].calc_loss(info)
        self._save_losses(active_agent_idx, loss_info)
        return loss_info

    def _save_losses(self, alg_idx: int, loss_info: LossInfo):
        """Extract and record actor and critic losses for a given algorithm."""
        if not hasattr(loss_info, 'extra') or loss_info.extra == ():
            return

        extra = loss_info.extra

        if hasattr(extra, 'critic'):
            critic_extra = extra.critic
            if critic_extra != () and isinstance(critic_extra, torch.Tensor):
                critic_loss_mean = critic_extra.mean().item()
                self._per_alg_critic_losses[alg_idx].append(
                    (self._train_step_counter, critic_loss_mean))

        if hasattr(extra, 'actor'):
            actor_extra = extra.actor
            if actor_extra != ():
                if hasattr(actor_extra,
                           'actor_loss') and actor_extra.actor_loss != ():
                    actor_loss_tensor = actor_extra.actor_loss
                    if isinstance(actor_loss_tensor, torch.Tensor):
                        actor_loss_mean = actor_loss_tensor.mean().item()
                        self._per_alg_actor_losses[alg_idx].append(
                            (self._train_step_counter, actor_loss_mean))

    def predict_step(self, inputs: TimeStep, state) -> AlgStep:
        batch_size = alf.nest.get_nest_size(inputs, dim=0)
        # During evaluation, batch_size may differ from training batch_size.
        # In that case, use the first sub-algorithm for prediction.
        if batch_size != self._batch_size:
            common.warning_once(
                f"predict_step batch_size mismatch: got {batch_size}, "
                f"expected {self._batch_size}. Using first sub-algorithm.")
            if isinstance(state, list) and len(state) == self._num_copies:
                alg_state = state[0]
            else:
                alg_state = self._algorithms[0].get_initial_predict_state(
                    batch_size)
            alg_step = self._algorithms[0].predict_step(inputs, alg_state)
            new_state = [alg_step.state] + [None] * (self._num_copies - 1)
            return alg_step._replace(state=new_state)

        sliced = self._slice_batch(inputs, state)
        results = {}
        new_states = [None] * self._num_copies
        for alg_idx, (
                sliced_time_step,
                sliced_state,
                batch_indices,
        ) in sliced.items():
            alg_step = self._algorithms[alg_idx].predict_step(
                sliced_time_step, sliced_state)
            new_states[alg_idx] = alg_step.state
            results[alg_idx] = (alg_step._replace(state=()), batch_indices)
        return scatter_and_sum_nested(
            results, self._batch_size)._replace(state=new_states)

    def _make_single_env(self):
        """Create a single environment using the configured env_load_fn and env_name."""
        env_load_fn = alf.get_config_value("create_environment.env_load_fn")
        env_name = alf.get_config_value("create_environment.env_name")
        return env_load_fn(env_name, env_id=0)

    def _call_debug_callback(self):
        """Call the debug callback if it exists."""
        if self._debug_callback is not None:
            self._debug_callback(replay_buffer=self._replay_buffer,
                                 algorithms=self._algorithms,
                                 action_spec=self._action_spec,
                                 num_copies=self._num_copies,
                                 iter_number=self._train_step_counter)

    def after_train_iter(self, inputs: TimeStep, info):
        assert alf.nest.get_nest_shape(inputs)[:2] == (
            self._unroll_length, self._env_counts
        ), f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

        sliced = self._slice_batch(inputs, time_major=True)
        for alg_idx, (
                sliced_inputs,
                _batch_indices,
        ) in sliced.items():
            self._algorithms[alg_idx].after_train_iter(sliced_inputs, info)

        self._call_debug_callback()
        self._train_step_counter += 1
        root_dir = self._config.root_dir if self._config else "."
        if (self._video_record_interval is not None and
                self._train_step_counter % self._video_record_interval == 0):
            video_dir = os.path.join(root_dir, "videos")
            self.record_videos(output_dir=video_dir,
                               num_episodes=1,
                               step_label=self._train_step_counter)
        if self._train_step_counter % self._return_logging_interval == 0:
            plot_dir = os.path.join(root_dir, "plots")
            self.save_ascii_plots(output_dir=plot_dir)
        if (self._agent_reset_period is not None
                and self._train_step_counter % self._agent_reset_period == 0):
            self._reset_agent(self._next_agent_to_reset)
            self._next_agent_to_reset = (self._next_agent_to_reset +
                                         1) % self._num_copies

    def after_update(self, root_inputs, info):
        assert alf.nest.get_nest_shape(root_inputs)[:2] == (
            self._mini_batch_length, self._batch_size
        ), f"root_inputs shape: {alf.nest.get_nest_shape(root_inputs)}"

        sliced = self._slice_batch(root_inputs, info, time_major=True)
        for alg_idx, (
                sliced_time_step,
                sliced_info,
                batch_indices,
        ) in sliced.items():
            self._algorithms[alg_idx].after_update(sliced_time_step,
                                                   sliced_info)

    @torch.no_grad()
    def record_videos(
        self,
        output_dir: str,
        env_ctor: Callable | None = None,
        num_episodes: int = 1,
        max_steps_per_episode: int = 1000,
        step_label: int | str = "",
        fps: int = 30,
        parallel: bool = True,
    ):
        """Record videos of each sub-agent's policy.

        Args:
            output_dir: Directory to save video files.
            env_ctor: Callable that creates a single environment instance.
                If None, uses the configured env_load_fn and env_name.
            num_episodes: Number of episodes to record per agent.
            max_steps_per_episode: Maximum steps per episode.
            step_label: Label to include in filename (e.g., training step).
            fps: Frames per second for output video.
            parallel: If True, record all agents in parallel using threads.
        """
        if env_ctor is None:
            env_ctor = self._make_single_env
        os.makedirs(output_dir, exist_ok=True)
        was_training = self.training
        self.eval()
        logging.info(
            f"Recording videos for {self._num_copies} agents (step={step_label})..."
        )

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x, dtype=torch.float32)

        def tensorize_time_step(ts):
            return ts._replace(
                observation=to_tensor(ts.observation),
                reward=to_tensor(ts.reward),
                step_type=torch.as_tensor(ts.step_type),
                discount=to_tensor(ts.discount),
            )

        def record_single_agent(agent_idx: int, alg):
            frames = []
            env = env_ctor()
            try:
                for ep in range(num_episodes):
                    env.reset()
                    time_step = tensorize_time_step(
                        common.get_initial_time_step(env))
                    state = alg.get_initial_predict_state(env.batch_size)
                    for _ in range(max_steps_per_episode):
                        frame = env.render(mode='rgb_array')
                        if frame is not None:
                            frames.append(frame)
                        is_first = time_step.is_first()
                        if not isinstance(is_first, torch.Tensor):
                            is_first = torch.tensor(is_first)
                        state = common.reset_state_if_necessary(
                            state,
                            alg.get_initial_predict_state(env.batch_size),
                            is_first)
                        alg_step = alg.predict_step(time_step, state)
                        state = alg_step.state
                        action = alg_step.output
                        if isinstance(action, torch.Tensor):
                            action = action.detach().cpu().numpy()
                        time_step = tensorize_time_step(env.step(action))
                        if time_step.is_last().any() if hasattr(
                                time_step.is_last(),
                                'any') else time_step.is_last():
                            break
            finally:
                env.close()

            if frames:
                filename = f"agent_{agent_idx}_{step_label}.mp4"
                output_path = os.path.join(output_dir, filename)
                common.save_video(frames, output_path, fps=fps)

        # On macOS, GLFW requires window creation on main thread - no parallel
        if parallel and sys.platform == 'darwin':
            logging.warning(
                "Disabling parallel video recording on macOS (GLFW main thread requirement)"
            )
            parallel = False

        if parallel:
            with ThreadPoolExecutor(max_workers=self._num_copies) as executor:
                futures = [
                    executor.submit(record_single_agent, i, alg)
                    for i, alg in enumerate(self._algorithms)
                ]
                for f in futures:
                    f.result()
        else:
            for i, alg in enumerate(self._algorithms):
                record_single_agent(i, alg)

        logging.info(f"Finished recording videos to {output_dir}")

        if was_training:
            self.train()

    def save_ascii_plots(self, output_dir: str):
        """Save ASCII plots of per-algorithm episode returns and losses."""
        from alf.utils.ascii_plotter import AsciiMetricPlotter
        os.makedirs(output_dir, exist_ok=True)

        returns_plotter = AsciiMetricPlotter(metrics_to_plot=[],
                                             smoothing_fraction=0.1)
        for alg_idx in sorted(self._per_alg_returns.keys()):
            returns = self._per_alg_returns[alg_idx]
            if returns:
                returns_plotter.set_history(f"return/{alg_idx}", returns)
        returns_plots = [
            returns_plotter.get_plot_string(name)
            for name in returns_plotter.get_metric_names()
        ]
        if returns_plots:
            path = os.path.join(output_dir, "episode_returns.txt")
            with open(path, "w") as f:
                f.write("\n\n".join(returns_plots))

        actor_plotter = AsciiMetricPlotter(metrics_to_plot=[],
                                           smoothing_fraction=0.1)
        for alg_idx in sorted(self._per_alg_actor_losses.keys()):
            actor_losses = self._per_alg_actor_losses[alg_idx]
            if actor_losses:
                actor_plotter.set_history(f"actor/{alg_idx}", actor_losses)
        actor_plots = [
            actor_plotter.get_plot_string(name, log_scale=True)
            for name in actor_plotter.get_metric_names()
        ]
        if actor_plots:
            path = os.path.join(output_dir, "actor_losses.txt")
            with open(path, "w") as f:
                f.write("\n\n".join(actor_plots))

        critic_plotter = AsciiMetricPlotter(metrics_to_plot=[],
                                            smoothing_fraction=0.1)
        for alg_idx in sorted(self._per_alg_critic_losses.keys()):
            critic_losses = self._per_alg_critic_losses[alg_idx]
            if critic_losses:
                critic_plotter.set_history(f"critic/{alg_idx}", critic_losses)
        critic_plots = [
            critic_plotter.get_plot_string(name, log_scale=True)
            for name in critic_plotter.get_metric_names()
        ]
        if critic_plots:
            path = os.path.join(output_dir, "critic_losses.txt")
            with open(path, "w") as f:
                f.write("\n\n".join(critic_plots))
