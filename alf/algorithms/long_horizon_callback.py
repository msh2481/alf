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
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from absl import logging
from concurrent.futures import ThreadPoolExecutor
import alf


@alf.configurable
class LongHorizonCallback:

    def __init__(self,
                 debug_env=None,
                 log_every_n_steps=100,
                 name="LongHorizonCallback"):
        self._debug_env = debug_env
        self._log_every_n_steps = log_every_n_steps
        self._debug_count = 0
        self._name = name
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _create_get_q_values_fn(self, algorithms, device):

        def get_q_values_fn(alg_index, obs, act):
            obs = obs.to(device).unsqueeze(0)
            act = act.to(device).unsqueeze(0)
            alg = algorithms[alg_index]
            q_values, _ = alg._compute_critics(alg._critic_networks,
                                               obs,
                                               act,
                                               critics_state=(),
                                               replica_min=True,
                                               apply_reward_weights=True)
            return q_values[0].item()

        return get_q_values_fn

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
        device = alf.get_default_device()
        get_q_values_fn = self._create_get_q_values_fn(algorithms, device)

        os.makedirs('logs', exist_ok=True)
        self._executor.submit(self._create_and_save_plots, iter_number,
                              num_copies, get_q_values_fn)
        logging.info(f"Plot saving in background")

    def _create_and_save_plots(self, iter_number, num_copies, get_q_values_fn):
        fig, axes = plt.subplots(num_copies,
                                 1,
                                 figsize=(10, 4 * num_copies),
                                 squeeze=False)

        for i in range(num_copies):
            ax = axes[i, 0]

            def q_func(obs, act):
                return get_q_values_fn(i, obs, act)

            obs_values, q_curves = self._debug_env.get_q_curves(q_func)

            for action, q_vals in q_curves.items():
                ax.plot(obs_values, q_vals, label=f'a={action}')

            ax.axvline(x=0.9,
                       color='gray',
                       linestyle='--',
                       alpha=0.5,
                       label='reward start')
            ax.set_xlabel('Observation (time progress)')
            ax.set_ylabel('Q-value')
            ax.set_title(f'Agent {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f'logs/{iter_number}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Written plot to {plot_path}")
