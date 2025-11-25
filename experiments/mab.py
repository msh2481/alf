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
"""Multi-Armed Bandit Experiments with Thompson Sampling variants."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
from tqdm import tqdm


class ArmStatistics:

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.mean_square = 0.0

    def update(self, reward: float) -> None:
        self.count += 1
        alpha = 1.0 / self.count
        self.mean += alpha * (reward - self.mean)
        self.mean_square += alpha * (reward**2 - self.mean_square)

    def get_mean(self) -> float:
        return self.mean

    def get_std(self) -> float:
        if self.count < 2:
            return 1.0
        variance = self.mean_square - self.mean**2
        variance = max(variance, 0)
        return np.sqrt(1 / self.count) + 1e-6

    def sample(self) -> float:
        return np.random.normal(self.get_mean(), self.get_std())


def thompson_sampling(arm_stats: List[ArmStatistics],
                      state: None = None) -> Tuple[int, None]:
    samples = [arm.sample() for arm in arm_stats]
    return int(np.argmax(samples)), None


def top_two_thompson_sampling(arm_stats: List[ArmStatistics],
                              play_first: bool) -> Tuple[int, bool]:
    first_arm = thompson_sampling(arm_stats)[0]

    if play_first:
        return first_arm, False
    else:
        second_arm = first_arm
        for _ in range(100):
            if second_arm == first_arm:
                break
            second_arm = thompson_sampling(arm_stats)[0]
        return second_arm, True


class GaussianBandit:

    def __init__(self, n_arms: int, means: np.ndarray, std: float = 0.1):
        self.n_arms = n_arms
        self.means = means
        self.std = std

    def pull(self, arm: int) -> float:
        return np.random.normal(self.means[arm], self.std)

    def get_optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    def get_optimal_value(self) -> float:
        return float(np.max(self.means))


def run_experiment(bandit: GaussianBandit,
                   algorithm: Callable,
                   n_steps: int,
                   init_state=None) -> Dict[str, np.ndarray]:
    arm_stats = [ArmStatistics() for _ in range(bandit.n_arms)]
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps)
    regrets = np.zeros(n_steps)
    optimal_value = bandit.get_optimal_value()
    state = init_state

    for step in range(n_steps):
        action, state = algorithm(arm_stats, state)
        reward = bandit.pull(action)
        arm_stats[action].update(reward)
        actions[step] = action
        rewards[step] = reward
        regrets[step] = optimal_value - reward

    # for i in range(bandit.n_arms):
    #     print(f"Arm {i}: {arm_stats[i].get_mean():.3f} ± {arm_stats[i].get_std():.3f}")
    # print()

    return {'actions': actions, 'rewards': rewards, 'regrets': regrets}


def run_verbose_experiment(n_arms: int, arm_std: float, n_steps: int) -> None:
    bandit = GaussianBandit(n_arms=n_arms,
                            means=np.random.randn(n_arms),
                            std=arm_std)

    print(f"\nBandit means: {bandit.means}")
    print(
        f"Optimal arm: {bandit.get_optimal_arm()} (mean: {bandit.get_optimal_value():.3f})"
    )

    ts_history = run_experiment(bandit, thompson_sampling, n_steps, None)
    tt_history = run_experiment(bandit, top_two_thompson_sampling, n_steps,
                                True)

    print(f"\nLast 30 actions:")
    print(f"Thompson Sampling:         {ts_history['actions'][-30:]}")
    print(f"Top-Two Thompson Sampling: {tt_history['actions'][-30:]}")

    print(f"\nFinal cumulative regrets:")
    print(f"Thompson Sampling:         {np.sum(ts_history['regrets']):.2f}")
    print(f"Top-Two Thompson Sampling: {np.sum(tt_history['regrets']):.2f}")


def run_multiple_experiments(n_arms: int, arm_std: float, algorithm: Callable,
                             init_state, n_steps: int,
                             n_runs: int) -> np.ndarray:
    all_regrets = np.zeros((n_runs, n_steps))

    for run in tqdm(range(n_runs)):
        bandit = GaussianBandit(n_arms=n_arms,
                                means=np.random.randn(n_arms),
                                std=arm_std)
        history = run_experiment(bandit, algorithm, n_steps, init_state)
        all_regrets[run] = history['regrets']

    return np.mean(all_regrets, axis=0)


def plot_results(results_dict: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for algo_name, regrets in results_dict.items():
        cumulative_regret = np.cumsum(regrets)
        ax1.plot(cumulative_regret, label=algo_name, alpha=0.8)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title('Cumulative Regret Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    window_size = 30
    for algo_name, regrets in results_dict.items():
        if len(regrets) >= window_size:
            smoothed_regret = np.convolve(regrets,
                                          np.ones(window_size) / window_size,
                                          mode='valid')
            ax2.plot(smoothed_regret, label=algo_name, alpha=0.8)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Smoothed Regret (window=100)')
    ax2.set_title('Smoothed Simple Regret')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    n_arms = 500
    n_steps = 2000
    n_runs = 50
    arm_std = 1.0

    print(
        f"Running {n_runs} experiments with {n_arms} arms for {n_steps} steps each..."
    )

    ts_regrets = run_multiple_experiments(n_arms, arm_std, thompson_sampling,
                                          None, n_steps, n_runs)

    tt_regrets = run_multiple_experiments(n_arms, arm_std,
                                          top_two_thompson_sampling, True,
                                          n_steps, n_runs)

    print(f"\nAverage final cumulative regrets:")
    print(f"  Thompson Sampling: {np.sum(ts_regrets):.2f}")
    print(f"  Top-Two Thompson Sampling: {np.sum(tt_regrets):.2f}")

    plot_results({
        'Thompson Sampling': ts_regrets,
        'Top-Two Thompson Sampling': tt_regrets
    })
    # run_verbose_experiment(n_arms, arm_std, n_steps)
