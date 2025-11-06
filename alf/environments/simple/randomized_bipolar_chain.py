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

import numpy as np
from alf.environments.simple.bipolar_chain import BipolarChain


class RandomizedBipolarChain(BipolarChain):
    """BipolarChain environment with pseudorandomized action effects.

    For each state, a random bit is generated (using a fixed seed for
    reproducibility). When this bit is 1, the effect of the action is
    flipped (XOR operation with the action).
    """

    def __init__(self, k=30):
        super().__init__(k)
        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(42)
        # Generate a random bit for each state (0 or 1)
        self.action_flip_bits = rng.randint(0, 2, size=self.num_states)

    def step(self, action):
        self.step_count += 1
        done = abs(self.state) >= self.k or self.step_count >= 2 * self.k
        if not done:
            # Get the flip bit for the current state
            flip_bit = self.action_flip_bits[self.state + self.k]
            # XOR the action with the flip bit to get the effective action
            effective_action = action ^ flip_bit
            # Apply the effective action to update state
            self.state += 2 * effective_action - 1
        reward = 1.0 if (self.state == self.k and not done) else 0.0
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[self.state + self.k] = 1.0
        return obs, reward, done, {}
