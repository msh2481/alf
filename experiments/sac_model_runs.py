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
    "PRIOR_SCALE": "1.0",
    "ALPHA": "0",
    "LN": "True",
    "UTD": "1",
    "RESET_PERIOD": "500",
    "NUM_AGENTS": "4",
    "NUM_ENVS": "4",
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
    "DYNAMICS_HIDDEN": "tuple()",
    "REWARD_HIDDEN": "(256,)",
    "SEEDS": "8",
    "BASE_DIR": "",
}

runs = {
    "baseline_sac": {
        "ALGO": "sac_grad",
        "GRAD_SYNC_WEIGHT": "0",
    },
    "sac_grad_1": {
        "ALGO": "sac_grad",
        "GRAD_SYNC_WEIGHT": "1",
    },
    "sac_grad_1e2": {
        "ALGO": "sac_grad",
        "GRAD_SYNC_WEIGHT": "1e2",
    },
    "sac_grad_1e3": {
        "ALGO": "sac_grad",
        "GRAD_SYNC_WEIGHT": "1e2",
    },
}

NAMES = list(runs.keys())


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

    plot_cmd = ("python tools/plot_sac_grad_diagnostics.py "
                f"--names {' '.join(NAMES)}")
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
