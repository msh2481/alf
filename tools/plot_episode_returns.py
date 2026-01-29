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
"""Interactive tool to plot IQM episode returns with confidence intervals.

Scans for events.ndjson files under an experiment root directory,
groups runs by name, and generates publication-quality plots with IQM, CI bands,
and quantile lines.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from absl import logging
from tqdm import tqdm

logging.set_verbosity(logging.INFO)


def discover_runs(
        root_dir: str,
        agents_as_seeds: bool = True,
        agent_reduce: str = "none") -> dict[str, dict[int, list[float]]]:
    root_path = Path(root_dir)
    if not root_path.exists():
        logging.warning(f"Root directory does not exist: {root_dir}")
        return {}

    groups: dict[str,
                 dict[int,
                      list[float]]] = defaultdict(lambda: defaultdict(list))

    for ndjson_file in root_path.rglob("events.ndjson"):
        rel_path = ndjson_file.relative_to(root_path)
        parts = rel_path.parts

        if len(parts) < 2:
            # expect <run_name>/events.ndjson
            logging.warning(f"Unexpected path structure: {rel_path}")
            continue
        run_name = parts[0]

        file_path = str(ndjson_file)

        if agents_as_seeds:
            episodes = extract_episodes(file_path)
            if agent_reduce == "max":
                # For max reduction, we need returns grouped by (episode_idx, agent_idx)
                # so we can take max across agents for each episode_idx
                raw_episodes = load_episode_returns(file_path)
                # Build dict: episode_idx -> {agent_idx -> return}
                per_agent: dict[int, dict[int, float]] = defaultdict(dict)
                for ep in raw_episodes:
                    eidx = ep["episode_idx"]
                    aidx = ep.get("agent_idx", 0)
                    per_agent[eidx][aidx] = ep["episode_return"]
                for episode_idx, agent_returns in per_agent.items():
                    if agent_returns:
                        groups[run_name][episode_idx].append(
                            max(agent_returns.values()))
            else:
                for episode_idx, returns in episodes.items():
                    groups[run_name][episode_idx].extend(returns)
        else:
            raw_episodes = load_episode_returns(file_path)
            agent_indices = set()
            for ep in raw_episodes:
                agent_indices.add(ep.get("agent_idx", 0))

            if not agent_indices:
                agent_indices = {0}

            for agent_idx in agent_indices:
                episodes = extract_episodes(file_path, agent_idx=agent_idx)
                group_name = f"{run_name}#{agent_idx}"
                for episode_idx, returns in episodes.items():
                    groups[group_name][episode_idx].extend(returns)

    return {k: dict(v) for k, v in groups.items()}


def load_episode_returns(file_path: str) -> list[dict]:
    episodes = []
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("type") == "episode":
                episodes.append(record)
    return episodes


def extract_episodes(file_path: str,
                     agent_idx: int | None = None) -> dict[int, list[float]]:
    episodes = load_episode_returns(file_path)
    result: dict[int, list[float]] = defaultdict(list)
    for ep in episodes:
        if agent_idx is not None and ep.get("agent_idx", 0) != agent_idx:
            continue
        episode_idx = ep["episode_idx"]
        episode_return = ep["episode_return"]
        result[episode_idx].append(episode_return)
    return result


def compute_iqm(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return float(stats.trim_mean(values, proportiontocut=0.25))


def bootstrap_ci(values: np.ndarray,
                 confidence: float = 0.95,
                 n_bootstrap: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    bootstrap_iqms = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_iqms.append(compute_iqm(sample))

    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_iqms, 100 * alpha / 2)
    upper = np.percentile(bootstrap_iqms, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def compute_statistics(
    episode_returns: dict[int, list[float]],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    episode_indices = []
    iqm_values = []
    ci_lows = []
    ci_highs = []
    q25_values = []
    q75_values = []

    max_episode = max(episode_returns.keys()) if episode_returns else 0
    num_trials = []

    for episode_idx in tqdm(range(1, max_episode + 1)):
        if episode_idx not in episode_returns:
            continue

        returns = episode_returns[episode_idx]
        num_trials.append(len(returns))

        returns_array = np.array(returns)
        iqm = compute_iqm(returns_array)
        ci_low, ci_high = bootstrap_ci(returns_array, confidence, n_bootstrap,
                                       bootstrap_seed)
        q25 = float(np.percentile(returns_array, 25))
        q75 = float(np.percentile(returns_array, 75))

        episode_indices.append(episode_idx)
        iqm_values.append(iqm)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        q25_values.append(q25)
        q75_values.append(q75)

    print(num_trials)

    return (np.array(episode_indices), np.array(iqm_values), np.array(ci_lows),
            np.array(ci_highs), np.array(q25_values), np.array(q75_values))


def interactive_select(groups: dict[str, dict[int, list[float]]]) -> list[str]:
    if not groups:
        logging.error("No run groups found!")
        return []

    group_names = sorted(groups.keys())
    selected = set(group_names)

    def print_status():
        print("\nSelect groups to plot:")
        for i, name in enumerate(group_names, 1):
            marker = "[x]" if name in selected else "[ ]"
            episode_data = groups[name]
            max_episode = max(episode_data.keys()) if episode_data else 0
            num_trials = sum(len(v) for v in episode_data.values())
            print(
                f"  {i}. {marker} {name}  (max_episode={max_episode}, num_trials={num_trials})"
            )
        print(
            "\nEnter numbers to toggle (space/comma-separated), 'all', 'none', or empty to proceed:"
        )

    print_status()

    while True:
        try:
            user_input = input().strip()
            if not user_input:
                break
            if user_input.lower() == "all":
                selected = set(group_names)
                print_status()
                continue
            if user_input.lower() == "none":
                selected = set()
                print_status()
                continue

            indices = []
            for part in re.split(r'[,\s]+', user_input):
                try:
                    idx = int(part)
                    if 1 <= idx <= len(group_names):
                        indices.append(idx - 1)
                except ValueError:
                    pass

            for idx in indices:
                name = group_names[idx]
                if name in selected:
                    selected.remove(name)
                else:
                    selected.add(name)

            print_status()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return []

    return sorted(selected)


def plot_groups(groups: dict[str, dict[int, list[float]]],
                selected: list[str],
                output_path: str,
                confidence: float = 0.95,
                n_bootstrap: int = 2000,
                bootstrap_seed: int = 0,
                max_episode: int | None = None,
                agent_reduce: str = "none"):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

    color_idx = 0
    for group_name in selected:
        episode_returns = groups[group_name]
        if not episode_returns:
            logging.warning(f"No episode data found for group: {group_name}")
            continue

        (episode_indices, iqm_values, ci_lows, ci_highs, q25_values,
         q75_values) = compute_statistics(episode_returns, confidence,
                                          n_bootstrap, bootstrap_seed)

        if max_episode is not None:
            mask = episode_indices <= max_episode
            episode_indices = episode_indices[mask]
            iqm_values = iqm_values[mask]
            ci_lows = ci_lows[mask]
            ci_highs = ci_highs[mask]
            q25_values = q25_values[mask]
            q75_values = q75_values[mask]

        if len(episode_indices) == 0:
            logging.warning(f"No valid episodes for group: {group_name}")
            continue

        color = colors[color_idx % len(colors)]
        color_idx += 1

        ax.plot(episode_indices,
                iqm_values,
                label=group_name,
                color=color,
                linewidth=2)
        ax.fill_between(episode_indices,
                        ci_lows,
                        ci_highs,
                        alpha=0.2,
                        color=color)
        ax.plot(episode_indices,
                q25_values,
                "--",
                color=color,
                linewidth=1,
                alpha=0.5)
        ax.plot(episode_indices,
                q75_values,
                "--",
                color=color,
                linewidth=1,
                alpha=0.5)

    ax.set_xlabel("Episode Index", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    title = "IQM Episode Return with 95% CI and Quantiles"
    if agent_reduce != "none":
        title += f" (agent_reduce={agent_reduce})"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot IQM episode returns with confidence intervals")
    parser.add_argument("--root_dir",
                        type=str,
                        default="/tmp/dmc",
                        help="Root directory to scan for runs")
    parser.add_argument("--out",
                        type=str,
                        default="iqm_episode_return.png",
                        help="Output PNG file path")
    parser.add_argument("--confidence",
                        type=float,
                        default=0.95,
                        help="Confidence level for CI (default: 0.95)")
    parser.add_argument("--n_bootstrap",
                        type=int,
                        default=1000,
                        help="Number of bootstrap samples (default: 1000)")
    parser.add_argument("--bootstrap_seed",
                        type=int,
                        default=0,
                        help="Random seed for bootstrap (default: 0)")
    parser.add_argument(
        "--agents_as_seeds",
        action="store_true",
        default=True,
        help="Treat each agent as a separate trial (default: True)")
    parser.add_argument(
        "--per_agent",
        action="store_true",
        default=False,
        help="Plot each agent as a separate line (default: False)")
    parser.add_argument(
        "--agent_reduce",
        type=str,
        choices=["none", "max"],
        default="none",
        help=("How to reduce the agent dimension before pooling trials. "
              "'none' pools all agents (default); 'max' takes pointwise max "
              "across agents per run/seed. Ignored with --per_agent."))
    parser.add_argument("--max_episode",
                        type=int,
                        default=None,
                        help="Maximum episode index to plot (default: None)")

    args = parser.parse_args()

    # Use per_agent mode if specified, otherwise use agents_as_seeds
    agents_as_seeds = not args.per_agent
    if args.per_agent and args.agent_reduce != "none":
        logging.warning("--agent_reduce is ignored when --per_agent is set.")

    logging.info(f"Scanning for runs under: {args.root_dir}")
    groups = discover_runs(args.root_dir,
                           agents_as_seeds=agents_as_seeds,
                           agent_reduce=args.agent_reduce)

    if not groups:
        logging.error(f"No runs found under {args.root_dir}")
        return

    logging.info(f"Found {len(groups)} run groups")
    selected = interactive_select(groups)

    if not selected:
        logging.warning("No groups selected, exiting")
        return

    logging.info(f"Plotting {len(selected)} groups...")
    plot_groups(groups,
                selected,
                args.out,
                args.confidence,
                args.n_bootstrap,
                args.bootstrap_seed,
                args.max_episode,
                agent_reduce=args.agent_reduce)


if __name__ == "__main__":
    main()
