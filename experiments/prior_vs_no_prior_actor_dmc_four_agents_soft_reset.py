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

from dm_control import suite
import subprocess

base = {
    "CONF": "experiments/dmc_actor_prior_conf.py",
    "LR": "1e-3",
    "WD": "1e-5",
    "GAMMA": "0.99",
    "ALPHA": "5e-4",
    "LN": "True",
    "UTD": "1",
    "RESET_PERIOD": "1",
    "NUM_AGENTS": "4",
    "NUM_ENVS": "4",
    "SCALE_BATCH": "True",
    "UNROLL_LENGTH": "1",
    "SHUFFLE": "False",
    "TAU": "0.01",
    "N_CRITICS": "1",
    "ASYNC": "False",
    "SHARE_ACTOR": "False",
    "SHARE_CRITIC": "True",
    "SHARED_CRITIC_MODE": "average",
    "OWN_ROLLOUT_FRACTION": "0.75",
    "NUM_LAYERS": "2",
    "SEEDS": "16",
    "BASE_DIR": "",
}
runs = {
    "a4_actorprior0": {
        "PRIOR_SCALE": "0.0"
    },
    "a4_actorprior0.001": {
        "PRIOR_SCALE": "0.001"
    },
    "a4_actorprior0.01": {
        "PRIOR_SCALE": "0.01"
    },
    "a4_actorprior0.1": {
        "PRIOR_SCALE": "0.1"
    },
    "a4_actorprior1.0": {
        "PRIOR_SCALE": "1.0"
    },
}
NAMES = list(runs.keys())
EPISODE_INDEX_BASE_AGENTS = 4


def pueue_add(command: str, after: list[str] | None = None) -> str:
    args = ["pueue", "add", "-p"]
    for task_id in (after or []):
        args += ["-a", task_id]
    args += ["--", command]
    return subprocess.run(args, check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> None:
    task_ids: list[str] = []
    for env in [f"{a}:{b}" for a, b in suite.BENCHMARKING]:
        for name, overrides in runs.items():
            p = {**base, "ENV": env, **overrides}
            args = " ".join(f'{k}="{v}"' for k, v in p.items())
            task_ids.append(
                pueue_add(f'bash scripts/run_dmc.sh {args} NAME="{name}"'))

    plot_cmd = ("python tools/custom_plot.py "
                f"--episode_index_base_agents {EPISODE_INDEX_BASE_AGENTS} "
                f"--names {' '.join(NAMES)}")
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
