from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import polars as pl

from plot_common import RunRef, discover_runs, iqm, resolve_folders

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
NAMES = ["a1_beta", "a4_beta", "a4_shuffle2"]
MAX_EPISODE: int | None = None
# Bin (episode_idx -> groups of consecutive episodes) before computing cummax.
# This mirrors the `n_bins` smoothing used in `tools/custom_plot.py`.
N_EPISODE_BINS = 50

Stat = Literal["iqm", "mean", "median"]
AgentReduce = Literal["none", "max", "mean"]


def _bin_mean_by_episode(
        t: np.ndarray,
        r: np.ndarray,
        *,
        n_bins: int,
        t_min: int | None = None,
        t_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin consecutive episode indices and average returns within each bin.

    Bins are defined by the episode span divided into ~n_bins chunks:
    bin = (episode_idx // bin_size) * bin_size, then mean within each bin.
    """
    if n_bins <= 0:
        return t, r
    if t.size == 0:
        return t, r

    if t_min is None:
        t_min = int(np.min(t))
    if t_max is None:
        t_max = int(np.max(t))
    span = max(1, int(t_max) - int(t_min) + 1)
    bin_size = max(1, int(np.ceil(span / n_bins)))
    t_bin = (t // bin_size) * bin_size

    order = np.argsort(t_bin, kind="mergesort")
    t_bin = t_bin[order]
    r = r[order]

    uniq, start = np.unique(t_bin, return_index=True)
    sums = np.add.reduceat(r, start)
    counts = np.diff(np.append(start, t_bin.size))
    means = sums / counts
    return uniq.astype(np.int64, copy=False), means.astype(np.float64, copy=False)


def _integral_one(t: np.ndarray, r: np.ndarray, *, smooth: bool = True) -> float | None:
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

    if smooth:
        # Smooth along episode axis (reduce noise) before cummax/integral.
        t, r = _bin_mean_by_episode(t, r, n_bins=N_EPISODE_BINS)
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


def _load_episode_last_returns(
        events_path: Path,
        *,
        max_episode: int | None,
) -> dict[int, dict[int, float]]:
    """Return per-agent per-episode last return.

    If multiple records exist for the same (agent_idx, episode_idx) within a
    single run file, we keep the *last* value (overwrite while streaming).
    """
    per_agent: dict[int, dict[int, float]] = defaultdict(dict)

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "episode":
                continue
            ep_idx = rec["episode_idx"]
            if max_episode is not None and ep_idx > max_episode:
                continue
            agent_idx = rec.get("agent_idx", 0) or 0
            per_agent[agent_idx][ep_idx] = rec["episode_return"]

    return dict(per_agent)


def _run_records(run: RunRef, *, agent_reduce_mode: AgentReduce,
                 max_episode: int | None) -> list[dict[str, object]]:
    per_agent = _load_episode_last_returns(run.events_path,
                                           max_episode=max_episode)
    if not per_agent:
        return []

    recs: list[dict[str, object]] = []

    if agent_reduce_mode == "none":
        for agent_idx, d in per_agent.items():
            if not d:
                continue
            t = np.fromiter(d.keys(), dtype=np.int64)
            r = np.fromiter(d.values(), dtype=np.float64)
            integral = _integral_one(t, r)
            se = None if integral is None else float(max(0.0, integral))
            recs.append({
                "experiment": run.experiment,
                "seed": run.seed,
                "agent_idx": agent_idx,
                "sample_eff": se,
            })
        return recs

    # Reduce across agents per episode_idx.
    if agent_reduce_mode == "max":
        # IMPORTANT: for max-across-agents, we smooth *per-agent* first, then
        # take max across agents on the smoothed (binned) episode axis.
        # This reduces noise before cummax and matches the "best agent per
        # timestep" ensemble interpretation more literally.
        t_min: int | None = None
        t_max: int | None = None

        cleaned: list[tuple[np.ndarray, np.ndarray]] = []
        for d in per_agent.values():
            if not d:
                continue
            t0 = np.fromiter(d.keys(), dtype=np.int64)
            r0 = np.fromiter(d.values(), dtype=np.float64)
            mask = np.isfinite(t0) & np.isfinite(r0) & (t0 > 0)
            t0 = t0[mask]
            r0 = r0[mask]
            if t0.size == 0:
                continue
            order = np.argsort(t0, kind="mergesort")
            t0 = t0[order]
            r0 = r0[order]
            cleaned.append((t0, r0))
            a = int(np.min(t0))
            b = int(np.max(t0))
            t_min = a if t_min is None else min(t_min, a)
            t_max = b if t_max is None else max(t_max, b)

        if not cleaned or t_min is None or t_max is None:
            return []

        ep_to_val: dict[int, float] = {}
        for t0, r0 in cleaned:
            t_b, r_b = _bin_mean_by_episode(t0,
                                            r0,
                                            n_bins=N_EPISODE_BINS,
                                            t_min=t_min,
                                            t_max=t_max)
            for ep_idx, v in zip(t_b.tolist(), r_b.tolist()):
                cur = ep_to_val.get(ep_idx)
                if cur is None or v > cur:
                    ep_to_val[ep_idx] = v
    else:  # mean
        ep_sum: dict[int, float] = {}
        ep_n: dict[int, int] = {}
        for d in per_agent.values():
            for ep_idx, v in d.items():
                ep_sum[ep_idx] = ep_sum.get(ep_idx, 0.0) + v
                ep_n[ep_idx] = ep_n.get(ep_idx, 0) + 1
        ep_to_val = {k: ep_sum[k] / ep_n[k] for k in ep_sum.keys()}

    if not ep_to_val:
        return []
    t = np.fromiter(ep_to_val.keys(), dtype=np.int64)
    r = np.fromiter(ep_to_val.values(), dtype=np.float64)
    # For max-across-agents we already smoothed per-agent before merging, so we
    # skip smoothing here to avoid binning twice.
    integral = _integral_one(t, r, smooth=(agent_reduce_mode != "max"))
    se = None if integral is None else float(max(0.0, integral))
    recs.append({
        "experiment": run.experiment,
        "seed": run.seed,
        "sample_eff": se,
    })
    return recs


def _per_run_metrics_for_env(*, folder: str, names: list[str],
                             agent_reduce_mode: AgentReduce,
                             max_episode: int | None,
                             workers: int) -> list[dict[str, object]]:
    runs = discover_runs(folder, names)
    print(f"Discovered {len(runs)} runs in {folder}.")
    if not runs:
        return []

    if workers <= 1:
        out: list[dict[str, object]] = []
        for run in runs:
            out.extend(
                _run_records(run,
                             agent_reduce_mode=agent_reduce_mode,
                             max_episode=max_episode))
        return out

    out: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for recs in ex.map(
                lambda rr: _run_records(rr,
                                        agent_reduce_mode=agent_reduce_mode,
                                        max_episode=max_episode), runs):
            out.extend(recs)
    return out


def _aggregate_by_experiment(records: list[dict[str, object]], *, env: str,
                             stat: Stat) -> pl.DataFrame:
    by_exp: dict[str, list[float]] = defaultdict(list)
    for r in records:
        exp = r.get("experiment")
        v = r.get("sample_eff")
        if not isinstance(exp, str):
            continue
        if v is None:
            continue
        by_exp[exp].append(float(v))

    rows: list[dict[str, object]] = []
    for exp, vals in by_exp.items():
        v = _reduce_stat(np.asarray(vals, dtype=np.float64), stat)
        rows.append({
            "experiment": exp,
            "env": env,
            "sample_eff": v,
            "n_runs": len(vals),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def compute_table(*,
                  folder_spec: str | Sequence[str] = FOLDER,
                  names: list[str] = NAMES,
                  stat: Stat = "iqm",
                  agent_reduce_mode: AgentReduce = "none",
                  max_episode: int | None = MAX_EPISODE,
                  workers: int = 1) -> pl.DataFrame:
    envs = resolve_folders(folder_spec)
    if not envs:
        return pl.DataFrame()

    all_rows: list[pl.DataFrame] = []
    env_order: list[str] = []
    for env_name, folder in envs:
        env_order.append(env_name)
        records = _per_run_metrics_for_env(folder=folder,
                                           names=names,
                                           agent_reduce_mode=agent_reduce_mode,
                                           max_episode=max_episode,
                                           workers=workers)
        all_rows.append(_aggregate_by_experiment(records, env=env_name,
                                                 stat=stat))

    long = pl.concat(all_rows, how="vertical") if all_rows else pl.DataFrame()
    if long.is_empty():
        return pl.DataFrame()

    wide = long.pivot(index="experiment",
                      on="env",
                      values="sample_eff",
                      aggregate_function="first")

    env_cols = [c for c in env_order if c in wide.columns]
    if not env_cols:
        return pl.DataFrame()
    wide = wide.select(["experiment", *env_cols])

    wide = wide.with_columns(
        pl.concat_list(env_cols).alias("_vals"),
    ).with_columns(
        pl.col("_vals").list.drop_nulls().list.eval(pl.element().log()
                                                    ).list.mean().exp().alias(
                                                        "average"),
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
        "How to handle agent_idx within each (seed,episode_idx).")
    p.add_argument("--max_episode",
                   type=int,
                   default=None,
                   help="Optional max episode_idx to include.")
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, (os.cpu_count() or 2)),
        help="Number of threads to read runs in parallel.")
    p.add_argument("--out",
                   type=str,
                   default="plots/sample_efficiency.tsv",
                   help=("Output path. The file will contain a fixed-width text "
                         "table (aligned columns). A second file "
                         "'sample_efficiency_max.tsv' will also be written in "
                         "the same folder, using pointwise max across agents "
                         "within each run/seed."))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    workers = max(1, int(args.workers))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Default table (as configured by --agent_reduce).
    table = compute_table(folder_spec=FOLDER,
                          names=NAMES,
                          stat=args.stat,
                          agent_reduce_mode=args.agent_reduce,
                          max_episode=args.max_episode,
                          workers=workers)
    if not table.is_empty():
        out_path.write_text(
            table.to_pandas().to_string(index=False, float_format="%.2f"),
            encoding="utf-8")

    # 2) "Ensemble" table: within each run/seed, take pointwise max across
    # agent dimension per timestep, then compute sample-efficiency on that
    # reduced curve.
    out_path_max = out_path.parent / "sample_efficiency_max.tsv"
    table_max = compute_table(folder_spec=FOLDER,
                              names=NAMES,
                              stat=args.stat,
                              agent_reduce_mode="max",
                              max_episode=args.max_episode,
                              workers=workers)
    if not table_max.is_empty():
        out_path_max.write_text(
            table_max.to_pandas().to_string(index=False, float_format="%.2f"),
            encoding="utf-8")


if __name__ == "__main__":
    main()

