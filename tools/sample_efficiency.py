from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from plot_common import (bin_agent_curves, discover_runs, load_episode_last_returns,
                         pointwise_agent_max, resolve_folders,
                         sample_efficiency_integral)

# Folder spec can be:
# - a single folder path (str)
# - a list/tuple of folder paths
# - the special string "all_dm", which expands to all subfolders of /tmp/dmc
FOLDER: str | Sequence[str] = [
    "/tmp/dmc/cartpole_swingup_sparse",
    "/tmp/dmc/fish_swim",
    "/tmp/dmc/cheetah_run",
    "/tmp/dmc/hopper_hop",
    "/tmp/dmc/hopper_stand",
    "/tmp/dmc/walker_run",
    "/tmp/dmc/walker_stand",
    "/tmp/dmc/walker_walk",
]
# NAMES = ["a4_shuffle2", "alpha_0.0005", "scale_0.001", "scale_0.003", "scale_0.01", "scale_0.03", "scale_0.1"]
NAMES = ["a4_shuffle2", "alpha_0.0005", "reset_1e2", "reset_1e3", "reset_3e3", "reset_1e4", "reset_1e5"]
MAX_EPISODE: int | None = None

# Fixed smoothing/binning on episode axis before cummax/integral.
EPISODE_BIN_SIZE = 10


def _load_binned_episode_curves(events_path: Path) -> dict[int, dict[int, float]]:
    """Load episode curves and immediately bin/smooth them."""
    per_agent = load_episode_last_returns(events_path, max_episode=MAX_EPISODE)
    return bin_agent_curves(per_agent, bin_size=EPISODE_BIN_SIZE)


def _records_for_env(
    *,
    folder: str,
    env_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return (per_agent_records, agent_max_records) for one env."""
    per_agent_records: list[dict[str, object]] = []
    agent_max_records: list[dict[str, object]] = []
    runs = discover_runs(folder, NAMES)
    print(f"Discovered {len(runs)} runs in {folder}.")
    for run in runs:
        per_agent = _load_binned_episode_curves(run.events_path)
        if not per_agent:
            continue

        # 1) Treat each (seed, agent_idx) curve as its own sample.
        for agent_idx, d in per_agent.items():
            if not d:
                continue
            t0 = np.fromiter(d.keys(), dtype=np.int64)
            r0 = np.fromiter(d.values(), dtype=np.float64)
            se = sample_efficiency_integral(t0, r0)
            if se is None:
                continue
            per_agent_records.append({
                "experiment": run.experiment,
                "env": env_name,
                "seed": run.seed,
                "agent_idx": int(agent_idx),
                "sample_eff": float(max(0.0, se)),
            })

        # 2) After binning, take pointwise max across agent dimension per seed.
        dmax = pointwise_agent_max(per_agent)
        if not dmax:
            continue
        t0 = np.fromiter(dmax.keys(), dtype=np.int64)
        r0 = np.fromiter(dmax.values(), dtype=np.float64)
        se = sample_efficiency_integral(t0, r0)
        if se is None:
            continue
        agent_max_records.append({
            "experiment": run.experiment,
            "env": env_name,
            "seed": run.seed,
            "sample_eff": float(max(0.0, se)),
        })

    return per_agent_records, agent_max_records


def _table_from_records(records: list[dict[str, object]],
                        env_order: list[str]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()

    long = pl.DataFrame(records).group_by(["experiment", "env"]).agg(
        pl.col("sample_eff").mean().alias("sample_eff"))

    wide = long.pivot(index="experiment",
                      on="env",
                      values="sample_eff",
                      aggregate_function="first")

    # Keep row order stable, include configs even if missing (null row).
    base = pl.DataFrame({"experiment": NAMES})
    wide = base.join(wide, on="experiment", how="left")

    env_cols = [c for c in env_order if c in wide.columns]
    if env_cols:
        wide = wide.select(["experiment", *env_cols])
        wide = wide.with_columns(
            pl.concat_list(env_cols).alias("_vals"),
        ).with_columns(
            pl.col("_vals").list.drop_nulls().list.eval(pl.element().log()
                                                           ).list.mean().exp().alias("average"),
        ).drop("_vals").sort("average", descending=True, nulls_last=True)
    else:
        wide = wide.with_columns(pl.lit(None).cast(pl.Float64).alias("average"))
    return wide


def main() -> None:
    envs = resolve_folders(FOLDER)
    if not envs:
        raise RuntimeError("No folders resolved from FOLDER")

    env_order = [name for name, _ in envs]

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    recs: list[dict[str, object]] = []
    recs_max: list[dict[str, object]] = []
    for env_name, folder in envs:
        a, b = _records_for_env(folder=folder, env_name=env_name)
        recs.extend(a)
        recs_max.extend(b)
    table = _table_from_records(recs, env_order)
    if not table.is_empty():
        (plots_dir / "sample_efficiency.tsv").write_text(
            table.to_pandas().to_string(index=False, float_format="%.2f"),
            encoding="utf-8")

    table_max = _table_from_records(recs_max, env_order)
    if not table_max.is_empty():
        (plots_dir / "sample_efficiency_max.tsv").write_text(
            table_max.to_pandas().to_string(index=False, float_format="%.2f"),
            encoding="utf-8")


if __name__ == "__main__":
    main()

