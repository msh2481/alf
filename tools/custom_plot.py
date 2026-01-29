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
"""Custom one-off plotting helper (hardcoded-first)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

FOLDER = "/tmp/dmc/Rotator"
NAMES = ["test-2"]
OUT = "iqm_episode_return.png"
MAX_EPISODE: int | None = None
CONFIDENCE = 0.95
N_BOOT = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class RunRef:
    experiment: str
    seed: str
    events_path: Path


def _discover_runs(folder: str, names: list[str]) -> list[RunRef]:
    root = Path(folder).expanduser()
    runs: list[RunRef] = []
    for exp in names:
        exp_dir = root / exp
        if not exp_dir.exists():
            continue
        for events_path in exp_dir.rglob("events.ndjson"):
            runs.append(
                RunRef(experiment=exp,
                       seed=events_path.parent.name,
                       events_path=events_path))
    return runs


def _iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_by_type(folder: str = FOLDER,
                 names: list[str] = NAMES) -> dict[str, pl.DataFrame]:
    runs = _discover_runs(folder, names)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for record in _iter_ndjson(run.events_path):
            event_type = record.get("type", "unknown")
            if not isinstance(event_type, str) or not event_type:
                event_type = "unknown"
            record = dict(record)
            record["experiment"] = run.experiment
            record["seed"] = run.seed
            grouped.setdefault(event_type, []).append(record)
    by_type: dict[str, pl.DataFrame] = {}
    for event_type, rows in grouped.items():
        by_type[event_type] = pl.DataFrame(rows) if rows else pl.DataFrame()
    return by_type


def agent_reduce(df: pl.DataFrame,
                 *,
                 x_col: str,
                 value_cols: Sequence[str],
                 reducer: Literal["max", "mean"] = "max",
                 group_cols: Sequence[str] = ("experiment", "seed"),
                 agent_col: str = "agent_idx") -> pl.DataFrame:
    """Reduce agent copies within each (group_cols + x_col).

    Example: max over agents per seed, then bootstrap across seeds.
    """
    if df.is_empty():
        return df
    if agent_col not in df.columns:
        return df

    if reducer == "max":
        agg_expr = lambda c: pl.col(c).max()
    elif reducer == "mean":
        agg_expr = lambda c: pl.col(c).mean()
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    keys = list(group_cols) + [x_col]
    aggs = [agg_expr(c).alias(c) for c in value_cols]
    return df.group_by(keys).agg(aggs)


def iqm(values: np.ndarray, proportion_to_cut: float = 0.25) -> float:
    """Interquartile mean (trimmed mean)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan")
    values.sort()
    k = int(np.floor(proportion_to_cut * n))
    if 2 * k >= n:
        return float(values.mean())
    return float(values[k:n - k].mean())


def bootstrap_ci(
    df: pl.DataFrame,
    *,
    group_cols: Sequence[str] = ("experiment", ),
    x_col: str,
    value_col: str,
    stat: Literal["iqm", "mean", "median"] = "iqm",
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Bootstrap CI per (group_cols + x_col), treating rows as samples."""
    if df.is_empty():
        return pl.DataFrame()
    if value_col not in df.columns:
        raise ValueError(f"Missing value_col: {value_col}")
    if x_col not in df.columns:
        raise ValueError(f"Missing x_col: {x_col}")

    if stat == "iqm":
        stat_fn: Callable[[np.ndarray], float] = iqm
    elif stat == "mean":
        stat_fn = lambda a: float(np.mean(a)) if a.size else float("nan")
    elif stat == "median":
        stat_fn = lambda a: float(np.median(a)) if a.size else float("nan")
    else:
        raise ValueError(f"Unknown stat: {stat}")

    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    keys = list(group_cols) + [x_col]

    for key_vals, g in df.group_by(keys, maintain_order=True):
        if len(keys) == 1:
            key_vals = (key_vals, )
        vals = g[value_col].to_numpy()
        vals = vals[np.isfinite(vals)]
        n = vals.size
        if n == 0:
            continue
        boots = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sample = rng.choice(vals, size=n, replace=True)
            boots[i] = stat_fn(sample)
        alpha = 1.0 - confidence
        lo = float(np.percentile(boots, 100 * (alpha / 2)))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        rec: dict[str, Any] = {k: v for k, v in zip(keys, key_vals)}
        rec.update({
            "n": int(n),
            "stat": float(stat_fn(vals)),
            "ci_low": lo,
            "ci_high": hi,
        })
        out.append(rec)

    return pl.DataFrame(out) if out else pl.DataFrame()


if __name__ == "__main__":
    by_type = load_by_type()

    for k, v in by_type.items():
        max_episode = v["episode_idx"].max(
        ) if "episode_idx" in v.columns else None
        print(k, v.shape, v["agent_idx"].unique().to_list(), max_episode)
        print(v.head().to_pandas().to_string())

    ep = by_type.get("episode", pl.DataFrame())
    if ep.is_empty():
        raise SystemExit("No episode events found")

    if MAX_EPISODE is not None:
        ep = ep.filter(pl.col("episode_idx") <= MAX_EPISODE)

    ci = bootstrap_ci(ep,
                      group_cols=("experiment", ),
                      x_col="episode_idx",
                      value_col="episode_return",
                      stat="iqm",
                      confidence=CONFIDENCE,
                      n_boot=N_BOOT,
                      seed=BOOTSTRAP_SEED)

    q = ep.group_by(["experiment", "episode_idx"]).agg(
        q25=pl.col("episode_return").quantile(0.25),
        q75=pl.col("episode_return").quantile(0.75),
    )

    plot_df = ci.join(q, on=["experiment", "episode_idx"],
                      how="left").sort(["experiment", "episode_idx"])

    fig, ax = plt.subplots(figsize=(10, 6))
    experiments = sorted(plot_df["experiment"].unique().to_list())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(experiments))))
    for i, exp in enumerate(experiments):
        d = plot_df.filter(pl.col("experiment") == exp).sort("episode_idx")
        x = d["episode_idx"].to_numpy()
        y = d["stat"].to_numpy()
        print(x.shape, y.shape)
        lo = d["ci_low"].to_numpy()
        hi = d["ci_high"].to_numpy()
        q25 = d["q25"].to_numpy()
        q75 = d["q75"].to_numpy()
        color = colors[i % len(colors)]
        ax.plot(x, y, label=exp, color=color, linewidth=2)
        ax.fill_between(x, lo, hi, alpha=0.2, color=color)
        ax.plot(x, q25, "--", color=color, linewidth=1, alpha=0.5)
        ax.plot(x, q75, "--", color=color, linewidth=1, alpha=0.5)

    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Episode Return")
    ax.set_title(f"IQM Episode Return with {int(CONFIDENCE * 100)}% CI")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {OUT}")
