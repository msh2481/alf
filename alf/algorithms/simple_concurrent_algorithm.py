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
import torch
import torch.nn as nn
import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.tensor_specs import TensorSpec
from alf.utils.common import slice_nested, scatter_and_sum_nested


@alf.configurable
class SimpleConcurrentAlgorithm(OffPolicyAlgorithm):

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
        name: str = "SimpleConcurrentAlgorithm",
        batch_size=None,
        env_counts=None,
        unroll_length=None,
        mini_batch_length=None,
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

        self._algorithms = nn.ModuleList([
            algorithm_ctor(
                observation_spec=observation_spec,
                action_spec=action_spec,
                reward_spec=reward_spec,
                env=None,
                config=config,
                debug_summaries=debug_summaries,
                name=f"{name}_copy_{i}",
            ) for i in range(num_copies)
        ])

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

    def train_step(self, inputs: TimeStep, state, rollout_info) -> AlgStep:
        total_batch_size = self._mini_batch_length * self._batch_size
        assert alf.nest.get_nest_size(
            inputs, dim=0
        ) == total_batch_size, f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

        sliced = self._slice_batch(inputs, state, rollout_info)
        results = {}
        new_states = [None] * self._num_copies
        for alg_idx, (
                sliced_time_step,
                sliced_state,
                sliced_rollout_info,
                batch_indices,
        ) in sliced.items():
            alg_step = self._algorithms[alg_idx].train_step(
                sliced_time_step, sliced_state, sliced_rollout_info)
            new_states[alg_idx] = alg_step.state
            results[alg_idx] = (alg_step._replace(state=()), batch_indices)
        return scatter_and_sum_nested(
            results, total_batch_size)._replace(state=new_states)

    def calc_loss(self, info) -> LossInfo:
        assert alf.nest.get_nest_shape(info)[:2] == (
            self._mini_batch_length,
            self._batch_size), f"info shape: {alf.nest.get_nest_shape(info)}"

        sliced = self._slice_batch(info, time_major=True)
        results = {}
        for alg_idx, (sliced_info, batch_indices) in sliced.items():
            results[alg_idx] = (
                self._algorithms[alg_idx].calc_loss(sliced_info),
                batch_indices)
        return scatter_and_sum_nested(results,
                                      self._batch_size,
                                      time_major=True)

    def predict_step(self, inputs: TimeStep, state) -> AlgStep:
        assert alf.nest.get_nest_size(
            inputs, dim=0
        ) == self._batch_size, f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

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

    def after_train_iter(self, inputs: TimeStep, info):
        assert alf.nest.get_nest_shape(inputs)[:2] == (
            self._unroll_length, self._env_counts
        ), f"inputs shape: {alf.nest.get_nest_shape(inputs)}"

        sliced = self._slice_batch(inputs)
        for alg_idx, (
                sliced_inputs,
                _batch_indices,
        ) in sliced.items():
            self._algorithms[alg_idx].after_train_iter(sliced_inputs, info)

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
