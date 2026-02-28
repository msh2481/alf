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
import math
import os
import json
import sys
import time
import torch
import torch.nn as nn
from absl import logging
import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.bipolar_callback import BipolarCallback
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.nest import slice_nested, scatter_and_sum_nested
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
        use_exploration_seeds: bool = False,
        use_parallel_training: bool = True,
        num_parallel_workers: Optional[int] = None,
        video_record_interval: int | None = None,
        return_logging_interval: int = 100,
        agent_reset_period: int | None = None,
        prior_perturbation_alpha: float | None = None,
        debug_env=None,
        debug_callback_cls=None,
        debug_log_every_n_steps: int = 100,
        log_states: bool = False,
        log_states_path: str | None = None,
        log_states_flush_interval: int = 100,
        log_episode_returns: bool = False,
        events_path: str | None = None,
        events_flush_interval: int = 10,
        log_losses: bool = True,
        log_weight_norms: bool = False,
        log_grad_norms: bool = False,
        diagnostics_logging_interval: int = 50,
        share_actor_across_copies: bool = False,
        shared_actor_mode: str = "first",
        share_critic_across_copies: bool = False,
        shared_critic_mode: str = "first",
        own_rollout_fraction: float = -1.0,
    ):

        self._batch_size = alf.get_config_value(
            "TrainerConfig.mini_batch_size")
        self._unroll_length = alf.get_config_value(
            "TrainerConfig.unroll_length")
        self._mini_batch_length = alf.get_config_value(
            "TrainerConfig.mini_batch_length")
        self._env_counts = alf.get_config_value(
            "create_environment.num_parallel_environments")
        assert self._batch_size % num_copies == 0, f"batch_size {self._batch_size} must be a multiple of num_copies {num_copies}"
        assert self._env_counts % num_copies == 0, f"env_counts {self._env_counts} must be a multiple of num_copies {num_copies}"
        try:
            shuffle_batch = alf.get_config_value("ReplayBuffer.shuffle_batch")
        except Exception:
            shuffle_batch = False
        assert not shuffle_batch, (
            "ConcurrentAlgorithm expects ReplayBuffer.shuffle_batch=False so "
            "env_id structure is preserved for per-copy routing.")

        temp_alg = algorithm_ctor(observation_spec=observation_spec,
                                  action_spec=action_spec,
                                  reward_spec=reward_spec)
        is_on_policy = temp_alg.on_policy
        del temp_alg
        train_state_spec = ()
        rollout_state_spec = ()
        predict_state_spec = ()

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

        # Episode return tracking (per env)
        self._per_env_cumulative_reward = torch.zeros(self._env_counts)
        self._per_env_episode_length = torch.zeros(self._env_counts,
                                                   dtype=torch.int32)
        self._per_alg_episode_counters: dict[int, int] = {
            i: 0
            for i in range(num_copies)
        }
        self._total_env_steps = 0

        # NDJSON events logging (episodes + losses)
        self._log_episode_returns = log_episode_returns
        self._log_losses = log_losses
        self._events_path = events_path
        self._events_flush_interval = max(1, events_flush_interval)
        self._events_file = None
        self._events_write_count = 0
        self._log_weight_norms = log_weight_norms
        self._log_grad_norms = log_grad_norms

        self._diagnostics_logging_interval = max(1,
                                                 diagnostics_logging_interval)
        self._share_actor_across_copies = share_actor_across_copies
        self._shared_actor_mode = shared_actor_mode
        self._share_critic_across_copies = share_critic_across_copies
        self._shared_critic_mode = shared_critic_mode
        if own_rollout_fraction is not None and own_rollout_fraction >= 0.0:
            assert own_rollout_fraction <= 1.0, (
                "own_rollout_fraction must be in [0, 1]")
            self._own_rollout_fraction = float(own_rollout_fraction)
        else:
            self._own_rollout_fraction = None
        self._last_train_batch_columns: Optional[dict[int,
                                                      torch.Tensor]] = None

        self._agent_reset_period = agent_reset_period
        self._next_agent_to_reset = 0
        self._prior_perturbation_alpha = prior_perturbation_alpha

        self._log_states = log_states
        self._log_states_path = log_states_path
        self._log_states_flush_interval = max(1, log_states_flush_interval)
        self._log_file = None
        self._rollout_step_counter = 0

        self._debug_callback = None
        if debug_env is not None and debug_callback_cls is not None:
            self._debug_callback = debug_callback_cls(
                debug_env=debug_env, log_every_n_steps=debug_log_every_n_steps)

    @torch.no_grad()
    def _build_shared_state(self, modules, mode_name: str):
        ref_state = modules[0].state_dict()
        shared_state = {}
        mode = mode_name.lower()
        if mode not in ("average", "first"):
            logging.warning(
                "Unsupported shared mode=%s. Fallback to 'average'.",
                mode_name)
            mode = "average"

        for k, ref_v in ref_state.items():
            if mode == "first":
                shared_state[k] = ref_v.detach().clone()
                continue

            if (not torch.is_tensor(ref_v) or not ref_v.dtype.is_floating_point
                    or ref_v.is_complex()):
                # Non-floating/bool/int buffers: use actor 0 values.
                shared_state[k] = ref_v.detach().clone()
                continue

            avg_v = torch.zeros_like(ref_v)
            for module in modules:
                avg_v.add_(module.state_dict()[k].detach())
            avg_v.div_(len(modules))
            shared_state[k] = avg_v

        return shared_state

    @torch.no_grad()
    def _sync_shared_actor(self):
        """Synchronize actor parameters across all copies.

        This provides an ablation where all copies share the same actor weights
        while keeping separate critic/alpha networks.
        """
        if not self._share_actor_across_copies or self._num_copies <= 1:
            return

        actors = []
        for alg in self._algorithms:
            actor = getattr(alg, "_actor_network", None)
            if actor is None:
                common.warning_once(
                    "share_actor_across_copies=True but sub-algorithm has no "
                    "_actor_network; actor sharing is skipped.")
                return
            actors.append(actor)

        shared_state = self._build_shared_state(actors,
                                                self._shared_actor_mode)

        for actor in actors:
            actor.load_state_dict(shared_state, strict=True)

    @torch.no_grad()
    def _sync_shared_critic(self):
        """Synchronize critic (and target critic) parameters across all copies."""
        if not self._share_critic_across_copies or self._num_copies <= 1:
            return

        critics = []
        target_critics = []
        for alg in self._algorithms:
            critic = getattr(alg, "_critic_networks", None)
            if critic is None:
                common.warning_once(
                    "share_critic_across_copies=True but sub-algorithm has no "
                    "_critic_networks; critic sharing is skipped.")
                return
            critics.append(critic)
            target_critic = getattr(alg, "_target_critic_networks", None)
            if target_critic is not None:
                target_critics.append(target_critic)

        shared_state = self._build_shared_state(critics,
                                                self._shared_critic_mode)
        for critic in critics:
            critic.load_state_dict(shared_state, strict=True)

        # Keep target critics aligned if they exist (e.g. SAC-like algorithms).
        if target_critics:
            shared_target_state = self._build_shared_state(
                target_critics, self._shared_critic_mode)
            for target_critic in target_critics:
                target_critic.load_state_dict(shared_target_state, strict=True)

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

    def _perturb_agent_prior(self, agent_idx: int):
        """Perturb randomized-prior networks inside an agent.

        This searches the sub-algorithm module tree for modules exposing
        ``perturb_prior(alpha)`` and calls it.
        """
        alpha = self._prior_perturbation_alpha
        if alpha is None:
            return

        alg = self._algorithms[agent_idx]
        perturbed = 0
        for module_name, m in alg.named_modules():
            # Keep target networks fixed during soft resets.
            is_target_module = ("_target_" in module_name
                                or module_name.startswith("target_")
                                or ".target_" in module_name)
            if is_target_module:
                continue
            perturb_fn = getattr(m, "perturb_prior", None)
            if callable(perturb_fn):
                perturb_fn(alpha)
                perturbed += 1

        if perturbed == 0:
            logging.warning(
                f"prior_perturbation_alpha is set but no perturbable modules were found in agent {agent_idx}; falling back to full reset"
            )
            self._reset_agent(agent_idx)

    def _apply_periodic_reset(self, agent_idx: int):
        """Apply periodic reset strategy for one agent.

        If ``prior_perturbation_alpha`` is positive, use soft reset by
        perturbing priors. Otherwise (None, 0, or negative), use full reset.
        """
        alpha = self._prior_perturbation_alpha
        if alpha is not None and alpha > 0:
            self._perturb_agent_prior(agent_idx)
        else:
            self._reset_agent(agent_idx)

    def close(self):
        """Clean up resources, including shutting down the thread pool executor."""
        if getattr(self, "_log_file", None) is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            finally:
                self._log_file = None
        if getattr(self, "_events_file", None) is not None:
            try:
                self._events_file.flush()
                self._events_file.close()
            finally:
                self._events_file = None
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

    def _slice_batch_with_indices(self,
                                  indices_by_alg: dict[int, torch.Tensor],
                                  *args,
                                  time_major=False):
        """Slice args with explicitly provided per-algorithm indices."""
        sliced = {}
        for i in range(self._num_copies):
            indices = indices_by_alg[i]
            sliced_args = []
            for arg in args:
                if isinstance(arg, list) and len(arg) == self._num_copies:
                    sliced_args.append(arg[i])
                else:
                    sliced_args.append(
                        slice_nested(arg, indices, time_major=time_major))
            sliced[i] = (*sliced_args, indices)
        return sliced

    @staticmethod
    def _stochastic_round(x: float, device) -> int:
        low = math.floor(x)
        frac = x - low
        if frac <= 0:
            return int(low)
        return int(low + (torch.rand((), device=device) < frac).item())

    def _make_train_batch_columns(self,
                                  inputs: TimeStep) -> dict[int, torch.Tensor]:
        """Build per-copy batch-column assignment with own-rollout bias.

        Assignment is defined over batch columns (size ``self._batch_size``) and
        reused across all time steps in the same update.
        """
        total = alf.nest.get_nest_size(inputs, dim=0)
        assert total % self._batch_size == 0, (
            f"train batch size {total} must be divisible by batch_size "
            f"{self._batch_size}")
        length = total // self._batch_size
        device = inputs.env_id.device
        env_ids = inputs.env_id.reshape(length,
                                        self._batch_size)[0].to(torch.int64)
        owners = torch.remainder(env_ids, self._num_copies)
        cols = torch.arange(self._batch_size, device=device)
        quota = self._batch_size // self._num_copies

        # Correct for accidental own samples from the shared pool.
        # Expected own fraction after random fill:
        #   f = 1/N + (1 - 1/N) * e
        # where N=num_copies and e is the explicit own-pick fraction.
        base_own = 1.0 / self._num_copies
        target_own = self._own_rollout_fraction
        if target_own <= base_own:
            explicit_own_fraction = 0.0
        else:
            explicit_own_fraction = ((target_own - base_own) /
                                     (1.0 - base_own))
            explicit_own_fraction = min(max(explicit_own_fraction, 0.0), 1.0)
        target_explicit_own = self._stochastic_round(
            explicit_own_fraction * quota, device)

        selected = {}
        needs = {}
        selected_mask = torch.zeros(self._batch_size,
                                    dtype=torch.bool,
                                    device=device)
        for i in range(self._num_copies):
            own_cols = cols[owners == i]
            take = min(target_explicit_own, own_cols.numel(), quota)
            if take > 0:
                perm = torch.randperm(own_cols.numel(), device=device)[:take]
                picked = own_cols[perm]
                selected_mask[picked] = True
            else:
                picked = torch.empty(0, dtype=torch.long, device=device)
            selected[i] = picked
            needs[i] = quota - take

        remaining = cols[~selected_mask]
        if remaining.numel() > 1:
            remaining = remaining[torch.randperm(remaining.numel(),
                                                 device=device)]

        cursor = 0
        batch_columns = {}
        for i in range(self._num_copies):
            need = needs[i]
            if need > 0:
                cols_i = torch.cat(
                    [selected[i], remaining[cursor:cursor + need]])
                cursor += need
            else:
                cols_i = selected[i]
            if cols_i.numel() > 1:
                cols_i = cols_i[torch.randperm(cols_i.numel(), device=device)]
            assert cols_i.numel() == quota, (
                f"copy {i} got {cols_i.numel()} columns, expected {quota}")
            batch_columns[i] = cols_i

        assert cursor == remaining.numel(), (
            "Unused columns remain after assignment")
        return batch_columns

    def _expand_train_batch_columns(self, batch_columns: dict[int,
                                                              torch.Tensor],
                                    length: int) -> dict[int, torch.Tensor]:
        """Expand per-copy batch columns to flattened [length * batch_size]."""
        device = next(iter(batch_columns.values())).device
        offsets = torch.arange(length,
                               device=device).unsqueeze(1) * self._batch_size
        flat_indices = {}
        for i in range(self._num_copies):
            flat_indices[i] = (offsets +
                               batch_columns[i].unsqueeze(0)).reshape(-1)
        return flat_indices

    def _ensure_log_file(self):
        if self._log_file is not None:
            return
        root_dir = self._config.root_dir if self._config else "."
        path = self._log_states_path or os.path.join(root_dir,
                                                     "rollout_states.ndjson")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._log_file = open(path, "a", encoding="utf-8")
        logging.info(f"Logging rollout states to {path}")

    def _ensure_events_file(self):
        if self._events_file is not None:
            return
        root_dir = self._config.root_dir if self._config else "."
        path = self._events_path or os.path.join(root_dir, "events.ndjson")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._events_file = open(path, "a", encoding="utf-8")
        logging.info(f"Logging events to {path}")

    def _write_event(self, record: dict):
        self._ensure_events_file()
        if self._events_file is None:
            return
        self._events_file.write(json.dumps(record) + "\n")
        self._events_write_count += 1
        if self._events_write_count % self._events_flush_interval == 0:
            self._events_file.flush()

    @staticmethod
    def _module_weight_norm(module: nn.Module | None) -> float | None:
        """Compute global L2 weight norm for a module (sqrt(sum(p^2)))."""
        if module is None:
            return None
        params = list(module.parameters())
        if not params:
            return None
        device = params[0].device
        total = torch.zeros((), device=device)
        for p in params:
            total = total + (p.detach()**2).sum()
        return float(torch.sqrt(total).item())

    @staticmethod
    def _module_grad_norm(module: nn.Module | None) -> float | None:
        """Compute global L2 grad norm for a module (sqrt(sum(g^2)))."""
        if module is None:
            return None
        params = list(module.parameters())
        if not params:
            return None
        device = params[0].device
        total = torch.zeros((), device=device)
        found = False
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total = total + (g**2).sum()
            found = True
        if not found:
            return None
        return float(torch.sqrt(total).item())

    def _to_jsonable(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        return value

    def _log_rollout_actions(self,
                             logged_alg_steps: dict[int,
                                                    tuple[AlgStep, TimeStep,
                                                          torch.Tensor]]):
        self._ensure_log_file()
        step = self._total_env_steps
        for alg_idx, (alg_step, time_step,
                      batch_indices) in logged_alg_steps.items():
            entry = {
                "step": step,
                "rollout_step": self._rollout_step_counter,
                "alg_idx": alg_idx,
                "env_indices": self._to_jsonable(batch_indices),
                "observation": self._to_jsonable(time_step.observation),
                "action": self._to_jsonable(alg_step.output),
                "episode_end": self._to_jsonable(time_step.is_last()),
            }
            self._log_file.write(json.dumps(entry) + "\n")
        if ((self._rollout_step_counter + 1) %
                self._log_states_flush_interval == 0):
            self._log_file.flush()

    def rollout_step(self, inputs: TimeStep, state) -> AlgStep:
        assert alf.nest.get_nest_size(
            inputs, dim=0
        ) == self._env_counts, f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

        # Track per-algorithm episode returns
        self._track_episode_returns(inputs)

        sliced = self._slice_batch(inputs, state)

        results = {}
        new_states = [None] * self._num_copies
        logged_alg_steps: dict[int, tuple[AlgStep, TimeStep,
                                          torch.Tensor]] = {}
        for alg_idx, (
                sliced_time_step,
                sliced_state,
                batch_indices,
        ) in sliced.items():
            alg_step = self._algorithms[alg_idx].rollout_step(
                sliced_time_step, sliced_state)
            new_states[alg_idx] = alg_step.state
            results[alg_idx] = (alg_step._replace(state=()), batch_indices)
            if self._log_states:
                logged_alg_steps[alg_idx] = (alg_step, sliced_time_step,
                                             batch_indices)

        if self._log_states and logged_alg_steps:
            self._log_rollout_actions(logged_alg_steps)
            self._rollout_step_counter += 1

        return scatter_and_sum_nested(
            results, self._env_counts)._replace(state=new_states)

    def _track_episode_returns(self, inputs: TimeStep):
        device = inputs.reward.device
        if self._per_env_cumulative_reward.device != device:
            self._per_env_cumulative_reward = self._per_env_cumulative_reward.to(
                device)
        if self._per_env_episode_length.device != device:
            self._per_env_episode_length = self._per_env_episode_length.to(
                device)
        self._total_env_steps += self._env_counts
        self._per_env_cumulative_reward += inputs.reward
        is_first = inputs.is_first()
        is_last = inputs.is_last()
        if is_first.any():
            for env_idx in is_first.nonzero(as_tuple=True)[0]:
                self._per_env_episode_length[env_idx] = 0
        self._per_env_episode_length += 1
        if is_last.any():
            for env_idx in is_last.nonzero(as_tuple=True)[0]:
                alg_idx = env_idx.item() % self._num_copies
                episode_return = self._per_env_cumulative_reward[env_idx].item(
                )
                episode_length = self._per_env_episode_length[env_idx].item()
                self._per_alg_episode_counters[alg_idx] += 1
                episode_idx = self._per_alg_episode_counters[alg_idx]
                logging.info(
                    f"Agent {alg_idx} episode {episode_idx} done: "
                    f"return={episode_return:.2f}, env_steps={self._total_env_steps}"
                )
                self._per_env_cumulative_reward[env_idx] = 0.0
                self._per_env_episode_length[env_idx] = 0
                if self._log_episode_returns:
                    episode_record = {
                        "type": "episode",
                        "agent_idx": alg_idx,
                        "env_idx": env_idx.item(),
                        "episode_idx": episode_idx,
                        "env_steps": self._total_env_steps,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "walltime": time.time(),
                    }
                    self._write_event(episode_record)

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
        if self._own_rollout_fraction is not None:
            self._last_train_batch_columns = self._make_train_batch_columns(
                inputs)
            flat_indices = self._expand_train_batch_columns(
                self._last_train_batch_columns, self._mini_batch_length)
            sliced = self._slice_batch_with_indices(flat_indices, inputs,
                                                    state, rollout_info)
        else:
            self._last_train_batch_columns = None
            sliced = self._slice_batch(inputs, state, rollout_info)

        def worker(alg_idx, sliced_data):
            """Worker function for agent training."""
            sliced_time_step, sliced_state, sliced_rollout_info, batch_indices = sliced_data
            alg_step = self._algorithms[alg_idx].train_step(
                sliced_time_step, sliced_state, sliced_rollout_info)
            return alg_idx, alg_step, batch_indices

        results = {}
        new_states = [None] * self._num_copies

        if self._use_parallel_training:
            # Parallel execution using ThreadPoolExecutor
            futures = [
                self._executor.submit(worker, alg_idx, sliced[alg_idx])
                for alg_idx in range(self._num_copies)
            ]
            for future in futures:
                alg_idx, alg_step, batch_indices = future.result()
                new_states[alg_idx] = alg_step.state
                results[alg_idx] = (alg_step._replace(state=()), batch_indices)
        else:
            # Sequential execution
            for alg_idx, sliced_data in sliced.items():
                alg_idx, alg_step, batch_indices = worker(alg_idx, sliced_data)
                new_states[alg_idx] = alg_step.state
                results[alg_idx] = (alg_step._replace(state=()), batch_indices)

        return scatter_and_sum_nested(
            results, total_batch_size)._replace(state=new_states)

    def calc_loss(self, info) -> LossInfo:
        assert alf.nest.get_nest_shape(info)[:2] == (
            self._mini_batch_length,
            self._batch_size), f"info shape: {alf.nest.get_nest_shape(info)}"
        if self._last_train_batch_columns is not None:
            sliced = self._slice_batch_with_indices(
                self._last_train_batch_columns, info, time_major=True)
        else:
            sliced = self._slice_batch(info, time_major=True)
        results = {}
        for alg_idx, (sliced_info, batch_indices) in sliced.items():
            loss_info = self._algorithms[alg_idx].calc_loss(sliced_info)
            results[alg_idx] = (loss_info, batch_indices)
            self._log_losses_event(alg_idx, loss_info)

        return scatter_and_sum_nested(results,
                                      self._batch_size,
                                      time_major=True)

    def _log_losses_event(self, alg_idx: int, loss_info: LossInfo):
        """Log flat scalar losses to events.ndjson if enabled."""
        if not self._log_losses:
            return
        train_iter = self._train_step_counter + 1
        if train_iter % self._diagnostics_logging_interval != 0:
            return
        extra = getattr(loss_info, 'extra', ())
        if extra == ():
            return

        actor_loss = None
        critic_loss = None
        alpha_loss = None
        critic_loss_nonfinite_frac = None
        actor_loss_nonfinite_frac = None

        def _float_or_none(v):
            if v == () or v is None:
                return None
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    return None
                v = float(v.detach().item())
            elif isinstance(v, (int, float)):
                v = float(v)
            else:
                return None
            return v if math.isfinite(v) else None

        # SAC (and some others) populate these extras.
        actor_extra = getattr(extra, 'actor', ())
        if actor_extra != ():
            actor_loss_tensor = getattr(actor_extra, 'actor_loss', ())
            if isinstance(actor_loss_tensor, torch.Tensor):
                actor_loss = actor_loss_tensor.mean().item()
                finite = torch.isfinite(actor_loss_tensor.detach())
                actor_loss_nonfinite_frac = (1.0 -
                                             finite.float().mean()).item()

        critic_extra = getattr(extra, 'critic', ())
        if isinstance(critic_extra, torch.Tensor):
            critic_loss = critic_extra.mean().item()
            finite = torch.isfinite(critic_extra.detach())
            critic_loss_nonfinite_frac = (1.0 - finite.float().mean()).item()

        # Common: alpha loss for SAC (extra.alpha is alpha loss tensor/scalar)
        alpha_extra = getattr(extra, 'alpha', ())
        if isinstance(alpha_extra, torch.Tensor):
            alpha_loss = alpha_extra.mean().item()
        elif isinstance(alpha_extra, (int, float)):
            alpha_loss = float(alpha_extra)

        if actor_loss is None and critic_loss is None:
            return

        record = {
            "type": "loss",
            "train_iter": self._train_step_counter + 1,
            "agent_idx": alg_idx,
            "env_steps": self._total_env_steps,
            "walltime": time.time(),
        }
        if actor_loss is not None:
            record["actor_loss"] = actor_loss
        if critic_loss is not None:
            record["critic_loss"] = critic_loss
        if alpha_loss is not None:
            record["alpha_loss"] = alpha_loss
        if actor_loss_nonfinite_frac is not None:
            v = _float_or_none(actor_loss_nonfinite_frac)
            if v is not None:
                record["actor_loss_nonfinite_frac"] = v
        if critic_loss_nonfinite_frac is not None:
            v = _float_or_none(critic_loss_nonfinite_frac)
            if v is not None:
                record["critic_loss_nonfinite_frac"] = v

        # SAC diagnostics (emitted only if present in extra)
        for k in (
                "log_alpha",
                "alpha_value",
                "log_pi_mean",
                "log_pi_min",
                "log_pi_max",
                "log_pi_nonfinite_frac",
                "entropy_reward_mean",
                "entropy_reward_min",
                "entropy_reward_max",
                "entropy_reward_nonfinite_frac",
                "target_q_mean",
                "target_q_min",
                "target_q_max",
                "target_q_nonfinite_frac",
        ):
            v = getattr(extra, k, ())
            sv = _float_or_none(v)
            if sv is not None:
                record[k] = sv
        self._write_event(record)

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
        assert alf.nest.get_nest_shape(inputs)[1] == self._env_counts, (
            f"inputs shape: {alf.nest.get_nest_shape(inputs)}")

        sliced = self._slice_batch(inputs, time_major=True)
        for alg_idx, (
                sliced_inputs,
                _batch_indices,
        ) in sliced.items():
            self._algorithms[alg_idx].after_train_iter(sliced_inputs, info)

        self._call_debug_callback()
        self._train_step_counter += 1

        if (self._log_weight_norms and self._train_step_counter %
                self._diagnostics_logging_interval == 0):
            for alg_idx, alg in enumerate(self._algorithms):
                actor_module = getattr(alg, "_actor_network", None)
                critic_module = getattr(alg, "_critic_networks", None)
                self._write_event({
                    "type": "weight_norm",
                    "train_iter": self._train_step_counter,
                    "agent_idx": alg_idx,
                    "env_steps": self._total_env_steps,
                    "walltime": time.time(),
                    "actor": self._module_weight_norm(actor_module),
                    "critic": self._module_weight_norm(critic_module),
                })

        if (self._log_grad_norms and self._train_step_counter %
                self._diagnostics_logging_interval == 0):
            for alg_idx, alg in enumerate(self._algorithms):
                actor_module = getattr(alg, "_actor_network", None)
                critic_module = getattr(alg, "_critic_networks", None)
                self._write_event({
                    "type": "grad_norm",
                    "train_iter": self._train_step_counter,
                    "agent_idx": alg_idx,
                    "env_steps": self._total_env_steps,
                    "walltime": time.time(),
                    "actor": self._module_grad_norm(actor_module),
                    "critic": self._module_grad_norm(critic_module),
                })

        root_dir = self._config.root_dir if self._config else "."
        if (self._video_record_interval is not None and
                self._train_step_counter % self._video_record_interval == 0):
            video_dir = os.path.join(root_dir, "videos")
            self.record_videos(output_dir=video_dir,
                               num_episodes=1,
                               step_label=self._train_step_counter)
        if (self._agent_reset_period is not None
                and self._train_step_counter % self._agent_reset_period == 0):
            self._apply_periodic_reset(self._next_agent_to_reset)
            self._next_agent_to_reset = (self._next_agent_to_reset +
                                         1) % self._num_copies

    def after_update(self, root_inputs, info):
        assert alf.nest.get_nest_shape(root_inputs)[:2] == (
            self._mini_batch_length, self._batch_size
        ), f"root_inputs shape: {alf.nest.get_nest_shape(root_inputs)}"
        if self._last_train_batch_columns is not None:
            sliced = self._slice_batch_with_indices(
                self._last_train_batch_columns,
                root_inputs,
                info,
                time_major=True)
        else:
            sliced = self._slice_batch(root_inputs, info, time_major=True)
        for alg_idx, (
                sliced_time_step,
                sliced_info,
                batch_indices,
        ) in sliced.items():
            self._algorithms[alg_idx].after_update(sliced_time_step,
                                                   sliced_info)

        # Optional ablation: force all copies to share the same actor weights.
        self._sync_shared_actor()
        # Optional ablation: force all copies to share the same critic weights.
        self._sync_shared_critic()
        self._last_train_batch_columns = None

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

        def to_tensor(x, device):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        def tensorize_time_step(ts, device):
            return ts._replace(
                observation=to_tensor(ts.observation, device),
                reward=to_tensor(ts.reward, device),
                step_type=torch.as_tensor(ts.step_type, device=device),
                discount=to_tensor(ts.discount, device),
            )

        def record_single_agent(agent_idx: int, alg):
            frames = []
            try:
                device = next(alg.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            env = env_ctor()
            try:
                for ep in range(num_episodes):
                    env.reset()
                    time_step = tensorize_time_step(
                        common.get_initial_time_step(env), device)
                    initial_state = alg.get_initial_predict_state(
                        env.batch_size)
                    initial_state = alf.nest.map_structure(
                        lambda x: x.to(device)
                        if isinstance(x, torch.Tensor) else x, initial_state)
                    state = initial_state
                    for _ in range(max_steps_per_episode):
                        frame = env.render(mode='rgb_array')
                        if frame is not None:
                            frames.append(frame)
                        is_first = time_step.is_first()
                        is_first = torch.as_tensor(is_first, device=device)
                        state = common.reset_state_if_necessary(
                            state, initial_state, is_first)
                        alg_step = alg.predict_step(time_step, state)
                        state = alg_step.state
                        action = alg_step.output
                        if isinstance(action, torch.Tensor):
                            action = action.detach().cpu().numpy()
                        time_step = tensorize_time_step(
                            env.step(action), device)
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
