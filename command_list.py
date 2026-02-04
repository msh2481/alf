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

names = [f"{a}:{b}" for a, b in suite.BENCHMARKING]
print(names)

for n in names:
    commands = [
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ENV="{n}" NAME="a4_shuffle"',
    ]
    for c in commands:
        print(f"pueue add -- '{c}'")
