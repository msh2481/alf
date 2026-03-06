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

import subprocess

base = {
    "LR": "1e-3",
    "WD": "1e-5",
    "GAMMA": "0.99",
    "PRIOR_SCALE": "0.01",
    "ALPHA": "0",
    "LN": "True",
    "UTD": "1",
    "RESET_PERIOD": "1e9",
    "NUM_AGENTS": "1",
    "NUM_ENVS": "1",
    "SCALE_BATCH": "True",
    "UNROLL_LENGTH": "1",
    "SHUFFLE": "False",
    "TAU": "0.05",
    "N_CRITICS": "1",
    "ASYNC": "False",
    "SHARE_ACTOR": "False",
    "SHARE_CRITIC": "False",
    "OWN_ROLLOUT_FRACTION": "-1",
    "NUM_LAYERS": "2",
    "MODEL_LOSS_WEIGHT": "1.0",
    "GRAD_SYNC_WEIGHT": "1.0",
    "SEEDS": "",
    "BASE_DIR": "",
}

runs = {
    "model_sac": {
        "ALGO": "sac",
    },
    "model_sac_v": {
        "ALGO": "sac_v",
    },
    "model_sac_grad": {
        "ALGO": "sac_grad",
    },
}

NAMES = list(runs.keys())
EPISODE_INDEX_BASE_AGENTS = 1


def pueue_add(command: str, after: list[str] | None = None) -> str:
    args = ["pueue", "add", "-p"]
    for task_id in (after or []):
        args += ["-a", task_id]
    args += ["--", command]
    return subprocess.run(args, check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> None:
    task_ids: list[str] = []
    for env in ["Rotator"]:
        for name, overrides in runs.items():
            p = {**base, "ENV": env, **overrides}
            args = " ".join(f'{k}="{v}"' for k, v in p.items())
            task_ids.append(
                pueue_add(
                    f'bash scripts/run_sac_model.sh {args} NAME="{name}"'))

    plot_cmd = ("python tools/custom_plot.py "
                f"--episode_index_base_agents {EPISODE_INDEX_BASE_AGENTS} "
                f"--names {' '.join(NAMES)}")
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
