from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class RunRef:
    experiment: str
    seed: str
    events_path: Path


def resolve_folders(folder_spec: str | Sequence[str]) -> list[tuple[str, str]]:
    """Resolve folder_spec into [(name, folder_path), ...].

    Special-case: if folder_spec contains "all_dm", expands to all subfolders of
    `/tmp/dmc`, sorted by name.
    """
    if isinstance(folder_spec, str):
        key = folder_spec.strip()
        if "all_dm" in key:
            dmc_root = Path("/tmp/dmc")
            folders = sorted([p for p in dmc_root.iterdir() if p.is_dir()],
                             key=lambda p: p.name)
            return [(p.name, str(p)) for p in folders]

        p = Path(folder_spec).expanduser()
        return [(p.name, str(p))]

    out: list[tuple[str, str]] = []
    for f in folder_spec:
        p = Path(f).expanduser()
        out.append((p.name, str(p)))
    return out


def discover_runs(folder: str, names: list[str]) -> list[RunRef]:
    """Discover runs under `<folder>/<experiment>/**/events.ndjson`."""
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


def iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_by_type(folder: str, names: list[str]) -> dict[str, pl.DataFrame]:
    """Load all records in discovered runs, grouped by record['type']."""
    runs = discover_runs(folder, names)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for record in iter_ndjson(run.events_path):
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


def load_type(folder: str,
              names: list[str],
              *,
              event_type: str) -> pl.DataFrame:
    runs = discover_runs(folder, names)
    rows: list[dict[str, Any]] = []
    for run in runs:
        for record in iter_ndjson(run.events_path):
            if record.get("type") != event_type:
                continue
            rec = dict(record)
            rec["experiment"] = run.experiment
            rec["seed"] = run.seed
            rows.append(rec)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def agent_reduce(df: pl.DataFrame,
                 *,
                 x_col: str,
                 value_cols: Sequence[str],
                 reducer: Literal["max", "mean"] = "max",
                 group_cols: Sequence[str] = ("experiment", "seed"),
                 agent_col: str = "agent_idx") -> pl.DataFrame:
    """Reduce agent copies within each (group_cols + x_col)."""
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

