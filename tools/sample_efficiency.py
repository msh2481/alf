from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import polars as pl

from plot_common import agent_reduce, iqm, load_type, resolve_folders

FOLDER: str | Sequence[str] = ["/tmp/dmc/acrobot_swingup", "/tmp/dmc/cartpole_swingup"]
NAMES = ["a1_beta", "a4_beta", "a4_shuffle2"]
MAX_EPISODE: int | None = None


Stat = Literal["iqm", "mean", "median"]
AgentReduce = Literal["none", "max", "mean"]


def _integral_one(t: np.ndarray, r: np.ndarray) -> float | None:
    """Compute I = ∑ (Δ cummax(r)) / t_at_increase for one run."""
    t = np.asarray(t)
    r = np.asarray(r)
    if t.size == 0 or r.size == 0:
        return None

    mask = np.isfinite(t) & np.isfinite(r)
    t = t[mask].astype(np.float64, copy=False)
    r = r[mask].astype(np.float64, copy=False)
    if t.size == 0:
        return None

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    r = r[order]

    # Episode index should be positive; drop non-positive to avoid 1/0.
    pos = t > 0
    t = t[pos]
    r = r[pos]
    if t.size == 0:
        return None

    r_cum = np.maximum.accumulate(r)
    dr = np.diff(r_cum)
    if dr.size == 0:
        return 0.0

    t2 = t[1:]
    inc = dr > 0
    if not np.any(inc):
        return 0.0
    return float(np.sum(dr[inc] / t2[inc]))


def _reduce_stat(values: np.ndarray, stat: Stat) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if stat == "iqm":
        v = iqm(values)
        return None if not np.isfinite(v) else float(v)
    if stat == "mean":
        return float(np.mean(values))
    if stat == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown stat: {stat}")


def _per_run_metrics(ep: pl.DataFrame, *, agent_reduce_mode: AgentReduce,
                     max_episode: int | None) -> pl.DataFrame:
    if ep.is_empty():
        return pl.DataFrame()
    if "episode_idx" not in ep.columns or "episode_return" not in ep.columns:
        return pl.DataFrame()

    ep = ep.select([c for c in ep.columns if c in {
        "experiment", "seed", "agent_idx", "episode_idx", "episode_return"
    }])

    ep = ep.with_columns(pl.col("episode_idx").cast(pl.Int64))
    if "agent_idx" in ep.columns:
        ep = ep.with_columns(pl.col("agent_idx").cast(pl.Int64))

    if max_episode is not None:
        ep = ep.filter(pl.col("episode_idx") <= max_episode)

    if ep.is_empty():
        return pl.DataFrame()

    if agent_reduce_mode in ("max", "mean"):
        ep = agent_reduce(ep,
                          x_col="episode_idx",
                          value_cols=("episode_return", ),
                          reducer=agent_reduce_mode,
                          group_cols=("experiment", "seed"),
                          agent_col="agent_idx")
        run_cols: list[str] = ["experiment", "seed"]
    else:
        run_cols = ["experiment", "seed"]
        if "agent_idx" in ep.columns:
            run_cols.append("agent_idx")

    # De-dup any repeated (run, episode_idx) records.
    ep = ep.group_by([*run_cols, "episode_idx"]).agg(
        pl.col("episode_return").mean().alias("episode_return"))

    out: list[dict[str, object]] = []
    for key_vals, g in ep.group_by(run_cols, maintain_order=True):
        if len(run_cols) == 1:
            key_vals = (key_vals, )
        rec = {k: v for k, v in zip(run_cols, key_vals)}
        t = g["episode_idx"].to_numpy()
        r = g["episode_return"].to_numpy()
        integral = _integral_one(t, r)
        if integral is None:
            se = None
        elif integral <= 0:
            # If the cumulative max never improves, I is 0; define SE=0.
            se = 0.0
        else:
            se = float(integral)
        rec["sample_eff"] = se
        out.append(rec)

    return pl.DataFrame(out) if out else pl.DataFrame()


def _aggregate_by_experiment(per_run: pl.DataFrame, *, env: str,
                             stat: Stat) -> pl.DataFrame:
    if per_run.is_empty():
        return pl.DataFrame(
            schema={
                "experiment": pl.String,
                "env": pl.String,
                "sample_eff": pl.Float64,
                "n_runs": pl.Int64,
            })

    rows: list[dict[str, object]] = []
    for exp_key, g in per_run.group_by("experiment", maintain_order=True):
        # Polars iteration returns tuples for group keys; normalize to scalar.
        exp = exp_key[0] if isinstance(exp_key, tuple) else exp_key
        vals = g["sample_eff"].to_numpy()
        v = _reduce_stat(vals, stat)
        rows.append({
            "experiment": exp,
            "env": env,
            "sample_eff": v,
            "n_runs": int(g.height),
        })
    return pl.DataFrame(rows)


def compute_table(*,
                  folder_spec: str | Sequence[str] = FOLDER,
                  names: list[str] = NAMES,
                  stat: Stat = "iqm",
                  agent_reduce_mode: AgentReduce = "none",
                  max_episode: int | None = MAX_EPISODE) -> pl.DataFrame:
    envs = resolve_folders(folder_spec)
    if not envs:
        return pl.DataFrame()

    all_rows: list[pl.DataFrame] = []
    env_order: list[str] = []
    for env_name, folder in envs:
        env_order.append(env_name)
        print(f"Loading episodes from {folder}...")
        ep = load_type(folder,
                       names,
                       event_type="episode")
        print(f"Loaded {ep.height} episodes from {folder}.")
        per_run = _per_run_metrics(ep,
                                   agent_reduce_mode=agent_reduce_mode,
                                   max_episode=max_episode)
        print(f"Computed {per_run.height} per-run metrics from {folder}.")
        all_rows.append(_aggregate_by_experiment(per_run, env=env_name,
                                                 stat=stat))

    long = pl.concat(all_rows, how="vertical") if all_rows else pl.DataFrame()
    if long.is_empty():
        return pl.DataFrame()

    wide = long.pivot(index="experiment",
                      on="env",
                      values="sample_eff",
                      aggregate_function="first")

    # Ensure stable env column ordering.
    env_cols = [c for c in env_order if c in wide.columns]
    wide = wide.select(["experiment", *env_cols])

    wide = wide.with_columns(
        pl.concat_list(env_cols).alias("_vals"),
        pl.col("experiment").cast(pl.String),
    ).with_columns(
        pl.col("_vals").list.drop_nulls().list.mean().alias("average"),
    ).drop("_vals")

    return wide.sort("average", descending=True, nulls_last=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute sample-efficiency table from events.ndjson logs.")
    p.add_argument("--stat",
                   type=str,
                   choices=["iqm", "mean", "median"],
                   default="mean",
                   help="How to aggregate per-run metrics per experiment.")
    p.add_argument(
        "--agent_reduce",
        type=str,
        choices=["none", "max", "mean"],
        default="none",
        help=
        "How to handle agent_idx within each (experiment,seed,episode_idx).")
    p.add_argument("--max_episode",
                   type=int,
                   default=None,
                   help="Optional max episode_idx to include.")
    p.add_argument("--out",
                   type=str,
                   default="plots/sample_efficiency.tsv",
                   help="Output path (TSV by default).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    table = compute_table(folder_spec=FOLDER,
                          names=NAMES,
                          stat=args.stat,
                          agent_reduce_mode=args.agent_reduce,
                          max_episode=args.max_episode)
    if table.is_empty():
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if out_path.suffix.lower() == ".csv" else "\t"
    table.write_csv(str(out_path), separator=sep)


if __name__ == "__main__":
    main()

