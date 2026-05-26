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

import gym
from gym import spaces
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

# BEZIER = np.array([0.0, 0.5, 1.2, -1.0, -0.5, 2.5, -0.5, -1.0, 1.0, 0.5, 0.0],
#                   dtype=np.float32)

C = 0.1322964213813880
BEZIER = np.array([0.0, -0.2, 1.1, -1.2, 1.0, -0.2, 0.0], dtype=np.float32) / C


class Bezier(gym.Env):

    def __init__(self, N=6):
        super().__init__()
        self.N = N
        assert len(
            BEZIER
        ) == N + 1, f"BEZIER must be an array of length N+1 ({N + 1}), got {len(BEZIER)}"
        self.bezier_coefs = BEZIER
        self.observation_space = spaces.Box(low=0.0,
                                            high=0.0,
                                            shape=(1, ),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.x = 0
        self.y = 0

    def reset(self):
        self.x = 0
        self.y = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.x += 1
        elif action == 1:
            self.y += 1

        reward = 0.0
        done = False

        if self.x + self.y == self.N:
            reward = self.bezier_coefs[self.x]
        elif self.x + self.y == self.N + 1:
            done = True

        obs = np.zeros(1, dtype=np.float32)
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        pass

    @staticmethod
    def show_curves():
        N = len(BEZIER) - 1

        def eval_prob(ps):
            ps = np.asarray(ps)
            x_vals = np.arange(N + 1)
            pn = binom.pmf(x_vals[:, None], N, ps[None, :])
            pnm1 = binom.pmf(x_vals[:-1, None], N - 1, ps[None, :])
            pr0 = np.zeros((N + 1, len(ps)))
            pr0[:N, :] = pnm1
            pr1 = np.zeros((N + 1, len(ps)))
            pr1[1:, :] = pnm1
            ex = np.sum(BEZIER[:, None] * pn, axis=0)
            ex0 = np.sum(BEZIER[:, None] * pr0, axis=0)
            ex1 = np.sum(BEZIER[:, None] * pr1, axis=0)
            return ex, ex0, ex1

        ps = np.linspace(0, 1, 300)
        ex, ex0, ex1 = eval_prob(ps)
        print(f"Max return: {np.max(ex)}")
        plt.figure(figsize=(14, 8))
        plt.plot(ps, ex, c="green", label='Overall')
        plt.plot(ps,
                 ex0,
                 lw=1,
                 linestyle="--",
                 c="blue",
                 label='After action 0')
        plt.plot(ps,
                 ex1,
                 lw=1,
                 linestyle="--",
                 c="purple",
                 label='After action 1')
        plt.xlabel('P(action=1)')
        plt.ylabel('Expected return')
        plt.legend()
        plt.ylim(-0.2, 1.2)
        # plt.subplot(2, 1, 2)
        # actor = np.exp(ex1) / (np.exp(ex0) + np.exp(ex1))
        # plt.plot(ps, actor, c="red", label='Next actor')
        # plt.plot(ps, ps, "--", c="blue", label='Current actor')
        # plt.xlabel('P(action=1)')
        # plt.ylabel('P(action=1)')
        # plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    Bezier.show_curves()
