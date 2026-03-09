#!/usr/bin/env python3
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
"""Custom regret-table helper mirroring custom_plot inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from plot_common import load_type, resolve_folders as _resolve_folders

# Keep defaults aligned with custom_plot.py without importing matplotlib there.
FOLDER: str | Sequence[str] = "all_dm"
NAMES = [
    "a4_prior0",
    "a4_prior0.001",
    "a4_prior0.01",
    "a4_prior0.1",
    "a4_prior1.0",
]
MAX_EPISODE: int | None = None
CORRECT_EPISODES = True
EPISODE_INDEX_BASE_AGENTS = 32
DEFAULT_WORKERS = 16


def _preprocess_episode(ep: pl.DataFrame,
                        *,
                        episode_index_base_agents: int) -> pl.DataFrame:
    if ep.is_empty() or "episode_idx" not in ep.columns:
        return ep

    x_col = "episode_idx"
    if CORRECT_EPISODES and "experiment" in ep.columns:
        base = float(episode_index_base_agents)
        num_agents = pl.col("experiment").cast(pl.String).str.extract(
            r"^a(\d+)(?:_|$)", 1).cast(pl.Float64, strict=False)
        scale = num_agents / base
        corrected_x = pl.when(scale.is_not_null() & (scale > 0)).then(
            pl.col(x_col).cast(pl.Float64) * scale).otherwise(
                pl.col(x_col).cast(pl.Float64))
        ep = ep.with_columns(corrected_x.round(8).alias(x_col))
    else:
        ep = ep.with_columns(pl.col(x_col).cast(pl.Int64))

    group_keys = ["experiment", "seed"]
    if "agent_idx" in ep.columns:
        group_keys.append("agent_idx")
    group_keys.append(x_col)

    numeric_cols: list[str] = []
    first_cols: list[str] = []
    for col_name, dtype in ep.schema.items():
        if col_name in group_keys:
            continue
        if dtype in (pl.Utf8, pl.String):
            first_cols.append(col_name)
        else:
            numeric_cols.append(col_name)

    aggs: list[pl.Expr] = []
    aggs.extend([pl.col(col_name).mean().alias(col_name) for col_name in numeric_cols])
    aggs.extend([pl.col(col_name).first().alias(col_name) for col_name in first_cols])
    return ep.group_by(group_keys, maintain_order=True).agg(aggs).sort(group_keys)


def _load_preprocessed_episode_df(*, folder: str, names: Sequence[str],
                                  max_episode: int | None,
                                  episode_index_base_agents: int) -> pl.DataFrame:
    ep = load_type(folder=folder, names=list(names), event_type="episode")
    if ep.is_empty():
        return ep
    if max_episode is not None and "episode_idx" in ep.columns:
        ep = ep.filter(pl.col("episode_idx") <= int(max_episode))
    return _preprocess_episode(
        ep, episode_index_base_agents=episode_index_base_agents)


def _reduce_episode_df(ep: pl.DataFrame,
                       *,
                       agent_reduce: str) -> pl.DataFrame:
    if ep.is_empty():
        return ep
    if agent_reduce == "none" or "agent_idx" not in ep.columns:
        return ep.sort(["experiment", "seed", "episode_idx"])

    keys = ["experiment", "seed", "episode_idx"]
    if agent_reduce == "mean":
        agg = pl.col("episode_return").mean().alias("episode_return")
    elif agent_reduce == "max":
        agg = pl.col("episode_return").max().alias("episode_return")
    else:
        raise ValueError(f"Unknown agent_reduce: {agent_reduce}")
    return ep.group_by(keys, maintain_order=True).agg(agg).sort(keys)


def _format_pm(mean: float | None, std: float | None) -> str:
    if mean is None or not np.isfinite(mean):
        return "-"
    if std is None or not np.isfinite(std):
        std = 0.0
    return f"{mean:.3f}±{std:.3f}"


def _format_average_pm(env_stats: pl.DataFrame, *, experiment: str) -> str:
    d = env_stats.filter(pl.col("experiment") == experiment)
    if d.is_empty():
        return "-"

    means = d["mean"].to_numpy().astype(np.float64, copy=False)
    stds = d["std"].to_numpy().astype(np.float64, copy=False)
    counts = d["n"].to_numpy().astype(np.float64, copy=False)

    mask = np.isfinite(means) & (means > 0) & np.isfinite(stds) & np.isfinite(
        counts) & (counts > 0)
    means = means[mask]
    stds = stds[mask]
    counts = counts[mask]
    if means.size == 0:
        return "-"

    avg_log = float(np.mean(np.log(means)))
    se_env = stds / np.sqrt(counts)
    log_var = np.square(se_env / means)
    se_log = float(np.sqrt(np.sum(log_var)) / means.size)

    geo_mean = float(np.exp(avg_log))
    geo_se = float(geo_mean * se_log)
    return _format_pm(geo_mean, geo_se)


def _warn_horizon_mismatch(*, env_name: str, horizons: pl.DataFrame) -> None:
    if horizons.is_empty():
        return
    min_h = horizons["max_episode_idx"].min()
    max_h = horizons["max_episode_idx"].max()
    if min_h is None or max_h is None or float(min_h) == float(max_h):
        return

    details = ", ".join(
        f"{row['experiment']}#{row['seed']}={row['max_episode_idx']:.1f}"
        for row in horizons.sort(["experiment", "seed"]).iter_rows(named=True))
    print(
        f"WARNING: horizon mismatch in {env_name}; "
        f"using shortest horizon {float(min_h):.1f} "
        f"(max {float(max_h):.1f}). {details}")


def _collect_horizon_mismatch_warning(*, env_name: str,
                                      horizons: pl.DataFrame) -> str | None:
    if horizons.is_empty():
        return None
    min_h = horizons["max_episode_idx"].min()
    max_h = horizons["max_episode_idx"].max()
    if min_h is None or max_h is None or float(min_h) == float(max_h):
        return None

    details = ", ".join(
        f"{row['experiment']}#{row['seed']}={row['max_episode_idx']:.1f}"
        for row in horizons.sort(["experiment", "seed"]).iter_rows(named=True))
    return (f"WARNING: horizon mismatch in {env_name}; "
            f"using shortest horizon {float(min_h):.1f} "
            f"(max {float(max_h):.1f}). {details}")


def _regret_records_from_episode_df(*, env_name: str, ep: pl.DataFrame,
                                    target_return: float,
                                    agent_reduce: str) -> tuple[
                                        list[dict[str, object]], list[str]]:
    messages: list[str] = []
    if not np.isfinite(target_return) or target_return <= 0:
        messages.append(
            f"WARNING: invalid target_return={target_return} for {env_name}; skipping."
        )
        return [], messages
    if ep.is_empty():
        messages.append(
            f"WARNING: no episode records found for {env_name}; skipping.")
        return [], messages

    ep = _reduce_episode_df(ep, agent_reduce=agent_reduce)
    if ep.is_empty():
        messages.append(
            f"WARNING: no reduced episode records found for {env_name}; skipping."
        )
        return [], messages

    horizons = ep.group_by(["experiment", "seed"], maintain_order=True).agg(
        pl.col("episode_idx").max().alias("max_episode_idx"))
    warning = _collect_horizon_mismatch_warning(env_name=env_name,
                                                horizons=horizons)
    if warning:
        messages.append(warning)

    common_horizon = horizons["max_episode_idx"].min()
    if common_horizon is None:
        messages.append(
            f"WARNING: could not determine common horizon for {env_name}; skipping."
        )
        return [], messages

    ep = ep.filter(pl.col("episode_idx") <= float(common_horizon)).with_columns(
        pl.when(pl.lit(float(target_return)) - pl.col("episode_return") > 0)
        .then(pl.lit(float(target_return)) - pl.col("episode_return"))
        .otherwise(pl.lit(0.0))
        .truediv(pl.lit(float(target_return)))
        .alias("regret_gap"))

    regret = ep.group_by(["experiment", "seed"], maintain_order=True).agg(
        pl.col("regret_gap").sum().alias("regret_gap_sum"),
        pl.len().alias("n_points"),
    ).with_columns(
        pl.when(pl.col("n_points") > 0).then(
            pl.col("regret_gap_sum") / pl.col("n_points").cast(pl.Float64))
        .otherwise(pl.lit(0.0)).alias("regret"))

    return ([{
        "env": env_name,
        "experiment": row["experiment"],
        "seed": row["seed"],
        "regret": float(row["regret"]),
        "horizon": float(common_horizon),
    } for row in regret.iter_rows(named=True)], messages)


def _summary_table(*, records: list[dict[str, object]], env_order: Sequence[str],
                   experiment_order: Sequence[str]) -> pl.DataFrame:
    if not records:
        rows = []
        for env_name in [*env_order, "Average"]:
            row = {"env": env_name}
            for experiment in experiment_order:
                row[experiment] = "-"
            rows.append(row)
        return pl.DataFrame(rows).select(["env", *experiment_order])

    df = pl.DataFrame(records)
    env_stats = df.group_by(["env", "experiment"], maintain_order=True).agg(
        pl.col("regret").mean().alias("mean"),
        pl.col("regret").std(ddof=1).alias("std"),
        pl.len().alias("n"),
    ).with_columns(pl.col("std").fill_null(0.0))

    row_maps: dict[str, dict[str, str]] = {}
    for env_name in env_order:
        row = {"env": env_name}
        env_df = env_stats.filter(pl.col("env") == env_name)
        stats = {
            rec["experiment"]: _format_pm(rec["mean"], rec["std"])
            for rec in env_df.iter_rows(named=True)
        }
        for experiment in experiment_order:
            row[experiment] = stats.get(experiment, "-")
        row_maps[env_name] = row

    avg_row = {"env": "Average"}
    for experiment in experiment_order:
        avg_row[experiment] = _format_average_pm(env_stats,
                                                 experiment=experiment)

    ordered_rows = [row_maps[env_name] for env_name in env_order]
    ordered_rows.append(avg_row)
    return pl.DataFrame(ordered_rows).select(["env", *experiment_order])


def _print_and_save_table(*, table: pl.DataFrame, out_root: Path,
                          stem: str) -> None:
    pd_table = table.to_pandas()
    print()
    print(pd_table.to_string(index=False))
    out_root.mkdir(parents=True, exist_ok=True)
    pd_table.to_csv(out_root / f"{stem}.tsv", sep="\t", index=False)
    (out_root / f"{stem}.txt").write_text(pd_table.to_string(index=False) + "\n",
                                          encoding="utf-8")
    print(f"Saved table to: {out_root / f'{stem}.tsv'}")


def _process_one_env(env_name: str, folder: str, names: Sequence[str],
                     max_episode: int | None, target_return: float,
                     episode_index_base_agents: int
                     ) -> tuple[str, list[dict[str, object]],
                                list[dict[str, object]], list[str]]:
    ep = _load_preprocessed_episode_df(
        folder=folder,
        names=names,
        max_episode=max_episode,
        episode_index_base_agents=episode_index_base_agents)
    mean_records, mean_messages = _regret_records_from_episode_df(
        env_name=env_name,
        ep=ep,
        target_return=target_return,
        agent_reduce="mean")
    max_records, max_messages = _regret_records_from_episode_df(
        env_name=env_name,
        ep=ep,
        target_return=target_return,
        agent_reduce="max")
    messages = list(dict.fromkeys([*mean_messages, *max_messages]))
    return env_name, mean_records, max_records, messages


def _build_tables(*, envs: Sequence[tuple[str, str]], names: Sequence[str],
                  max_episode: int | None,
                  target_return: float,
                  episode_index_base_agents: int,
                  workers: int,
                  ) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_records_mean: list[dict[str, object]] = []
    all_records_max: list[dict[str, object]] = []
    total = len(envs)
    if workers <= 1 or total <= 1:
        for idx, (env_name, folder) in enumerate(envs, start=1):
            print(f"\n=== [{idx}/{total}] Processing {env_name} ({folder}) ===")
            _, mean_records, max_records, messages = _process_one_env(
                env_name=env_name,
                folder=folder,
                names=names,
                max_episode=max_episode,
                target_return=target_return,
                episode_index_base_agents=episode_index_base_agents)
            for message in messages:
                print(message)
            all_records_mean.extend(mean_records)
            all_records_max.extend(max_records)
    else:
        print(f"Using {workers} worker processes across {total} envs.")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(
                _process_one_env,
                [env_name for env_name, _ in envs],
                [folder for _, folder in envs],
                [list(names)] * total,
                [max_episode] * total,
                [target_return] * total,
                [episode_index_base_agents] * total,
            )
            for idx, (env_name, folder) in enumerate(envs, start=1):
                result_env_name, mean_records, max_records, messages = next(
                    results)
                print(
                    f"\n=== [{idx}/{total}] Processed {result_env_name} ({folder}) ==="
                )
                for message in messages:
                    print(message)
                all_records_mean.extend(mean_records)
                all_records_max.extend(max_records)
    env_order = [env_name for env_name, _ in envs]
    return (_summary_table(records=all_records_mean,
                           env_order=env_order,
                           experiment_order=names),
            _summary_table(records=all_records_max,
                           env_order=env_order,
                           experiment_order=names))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cumulative regret tables from experiment logs.")
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help=
        "Experiment names to include (space-separated). Defaults to hardcoded NAMES.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Root folder containing experiment runs. Defaults to hardcoded FOLDER.",
    )
    parser.add_argument(
        "--episode_index_base_agents",
        type=int,
        default=EPISODE_INDEX_BASE_AGENTS,
        help=("Base number of agents used for episode-index x-axis scaling "
              f"(default: {EPISODE_INDEX_BASE_AGENTS})."),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots",
        help="Output directory for saved tables.",
    )
    parser.add_argument(
        "--max_episode",
        type=int,
        default=MAX_EPISODE,
        help="Optional maximum corrected episode index to keep.",
    )
    parser.add_argument(
        "--target_return",
        type=float,
        default=1000.0,
        help="Upper-bound return used to compute cumulative regret.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=
        f"Number of worker processes across envs. Defaults to {DEFAULT_WORKERS} on this machine.",
    )
    args = parser.parse_args()

    names = args.names or NAMES
    episode_index_base_agents = int(args.episode_index_base_agents)

    folder = args.folder or FOLDER
    envs = _resolve_folders(folder)
    if not envs:
        raise RuntimeError("No folders resolved from FOLDER")

    out_root = Path(args.out_dir)
    workers = args.workers
    if workers is None:
        workers = min(len(envs), DEFAULT_WORKERS)
    workers = max(1, int(workers))

    table_mean, table_max = _build_tables(envs=envs,
                                          names=names,
                                          max_episode=args.max_episode,
                                          target_return=float(
                                              args.target_return),
                                          episode_index_base_agents=
                                          episode_index_base_agents,
                                          workers=workers)
    print("\n=== Regret Table (agent-mean) ===")
    _print_and_save_table(table=table_mean,
                          out_root=out_root,
                          stem="regret_mean")

    print("\n=== Regret Table (agent-max) ===")
    _print_and_save_table(table=table_max,
                          out_root=out_root,
                          stem="regret_max")


if __name__ == "__main__":
    main()
