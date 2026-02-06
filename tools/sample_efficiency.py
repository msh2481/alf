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
        ep_to_val: dict[int, float] = {}
        for d in per_agent.values():
            for ep_idx, v in d.items():
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
    integral = _integral_one(t, r)
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
                   help="Output path (TSV by default; .csv uses comma).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    workers = max(1, int(args.workers))
    table = compute_table(folder_spec=FOLDER,
                          names=NAMES,
                          stat=args.stat,
                          agent_reduce_mode=args.agent_reduce,
                          max_episode=args.max_episode,
                          workers=workers)
    if table.is_empty():
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if out_path.suffix.lower() == ".csv" else "\t"
    table.write_csv(str(out_path), separator=sep, float_precision=2)


if __name__ == "__main__":
    main()

