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

# names = [f"{a}:{b}" for a, b in suite.BENCHMARKING]
# print(names)

EXECUTE = True

names = [
    "cartpole:swingup_sparse",
    "fish:swim",
    "cheetah:run",
    "hopper:hop",
    "hopper:stand",
    "walker:run",
    "walker:stand",
    "walker:walk",
]

def pueue_add(command):
    if EXECUTE:
        subprocess.run(["pueue", "add", "--", command], check=True)
    else:
        print(f"pueue add -- '{command}'")

for n in names:
    commands = [
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ALPHA=0.0 RESET_PERIOD=100 ENV="{n}" NAME="reset_1e2"',
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ALPHA=0.0 RESET_PERIOD=1000 ENV="{n}" NAME="reset_1e3"',
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ALPHA=0.0 RESET_PERIOD=3000 ENV="{n}" NAME="reset_3e3"',
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ALPHA=0.0 RESET_PERIOD=10000 ENV="{n}" NAME="reset_1e4"',
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ALPHA=0.0 RESET_PERIOD=100000 ENV="{n}" NAME="reset_1e5"',
    ]
    for c in commands:
        pueue_add(c)

pueue_add("python tools/sample_efficiency.py")
pueue_add("python tools/custom_plot.py")
