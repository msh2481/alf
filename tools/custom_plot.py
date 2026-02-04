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

# FOLDER = "/tmp/dmc/Rotator"
# FOLDER = "/tmp/dmc/pendulum_swingup"
# FOLDER can be:
# - a single folder path (str)
# - a list/tuple of folder paths
# - the special string "all_dm", which expands to all
#   subfolders of /tmp/dmc.
FOLDER: str | Sequence[str] = "all_dm"
NAMES = ["a1", "a4", "a1_beta", "a4_beta"]
OUT = "iqm_episode_return.png"
OUT_LINES = "lines_episode_return.png"
OUT_CRITIC = "critic.png"
MAX_EPISODE: int | None = None
CONFIDENCE = 0.95
N_BOOT = 100
BOOTSTRAP_SEED = 0
N_BINS = 50


@dataclass(frozen=True)
class RunRef:
    experiment: str
    seed: str
    events_path: Path


def _resolve_folders(
        folder_spec: str | Sequence[str]) -> list[tuple[str, str]]:
    """Resolve FOLDER into [(name, folder_path), ...]."""
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


def episode_plot_df(ep: pl.DataFrame,
                    *,
                    group_cols: Sequence[str] = ("experiment", ),
                    max_episode: int | None = None,
                    confidence: float = 0.95,
                    n_boot: int = 2000,
                    bootstrap_seed: int = 0,
                    n_bins: int | None = None) -> pl.DataFrame:
    if ep.is_empty():
        return pl.DataFrame()
    if max_episode is not None:
        ep = ep.filter(pl.col("episode_idx") <= max_episode)
    if not ep.is_empty() and "episode_idx" in ep.columns:
        ep = ep.with_columns(pl.col("episode_idx").cast(pl.Int64))

    # Optional binning along episode axis. Important for long runs where the
    # per-episode curve is too dense/noisy.
    x_col = "episode_idx"
    if n_bins is not None and n_bins > 0:
        x_min = ep.select(pl.col(x_col).min()).item()
        x_max = ep.select(pl.col(x_col).max()).item()
        if x_min is not None and x_max is not None:
            x_min = int(x_min)
            x_max = int(x_max)
            span = max(1, x_max - x_min + 1)
            bin_size = max(1, int(np.ceil(span / n_bins)))
            x_bin_col = f"{x_col}_bin"
            ep = ep.with_columns(
                (pl.col(x_col) // bin_size * bin_size).alias(x_bin_col))
            x_col = x_bin_col

            # Reduce within each (seed[,agent],bin) so bootstrap treats each
            # curve as one sample per x-bin (similar spirit to train_iter binning).
            reduce_keys = list(group_cols) + ["seed", x_col]
            if "agent_idx" in ep.columns:
                reduce_keys.insert(len(group_cols) + 1, "agent_idx")
            ep = ep.group_by(reduce_keys).agg(
                pl.col("episode_return").mean().alias("episode_return"))

    ci = bootstrap_ci(ep,
                      group_cols=group_cols,
                      x_col=x_col,
                      value_col="episode_return",
                      stat="iqm",
                      confidence=confidence,
                      n_boot=n_boot,
                      seed=bootstrap_seed)
    q = ep.group_by([*group_cols, x_col]).agg(
        q25=pl.col("episode_return").quantile(0.25),
        q75=pl.col("episode_return").quantile(0.75),
    )
    return ci.join(q, on=[*group_cols, x_col],
                   how="left").sort([*group_cols, x_col])


def plot_episode_iqm(plot_df: pl.DataFrame,
                     *,
                     out: str,
                     group_col: str = "experiment",
                     confidence: float = 0.95):
    if plot_df.is_empty() or group_col not in plot_df.columns:
        print(
            f"Empty plot_df or missing '{group_col}'; skipping IQM plot: {out}"
        )
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(plot_df[group_col].unique().to_list())
    colors = plt.cm.Set1(0.05 + 0.1 * np.arange(max(1, len(groups))))
    for i, g in enumerate(groups):
        # The x column can be either episode_idx or episode_idx_bin depending on
        # whether binning is enabled upstream.
        x_col = "episode_idx" if "episode_idx" in plot_df.columns else "episode_idx_bin"
        d = plot_df.filter(pl.col(group_col) == g).sort(x_col)
        x = d[x_col].to_numpy()
        y = d["stat"].to_numpy()
        lo = d["ci_low"].to_numpy()
        hi = d["ci_high"].to_numpy()
        q25 = d["q25"].to_numpy()
        q75 = d["q75"].to_numpy()
        color = colors[i % len(colors)]
        ax.plot(x, y, label=g, color=color, linewidth=1)
        ax.fill_between(x, lo, hi, alpha=0.2, color=color)
        ax.plot(x, q25, "--", color=color, linewidth=1, alpha=0.5)
        ax.plot(x, q75, "--", color=color, linewidth=1, alpha=0.5)
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Episode Return")
    ax.set_title(f"IQM Episode Return with {int(confidence * 100)}% CI")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")


def plot_many_lines(ax,
                    df: pl.DataFrame,
                    *,
                    x_col: str,
                    y_col: str,
                    line_cols: Sequence[str] = ("seed", "agent_idx"),
                    color_col: str = "experiment",
                    title: str | None = None,
                    alpha: float = 0.25,
                    linewidth: float = 1.0,
                    linestyle: str = "-",
                    n_bins: int | None = None,
                    bin_reducer: Literal["mean"] = "mean",
                    set_ylim_quantiles: bool = True,
                    ylim_quantiles: tuple[float, float] = (0.05, 0.95),
                    yscale: Literal["linear", "log", "symlog"] = "linear",
                    symlog_linthresh: float = 1.0,
                    show_legend: bool = True):
    if df.is_empty() or x_col not in df.columns or y_col not in df.columns:
        ax.set_axis_off()
        return

    # Some event types might not have all requested line columns (e.g. no
    # `agent_idx`). In that case, gracefully drop missing ones.
    line_cols = tuple(c for c in line_cols if c in df.columns)

    if n_bins is not None and n_bins > 0:
        x_min = df.select(pl.col(x_col).min()).item()
        x_max = df.select(pl.col(x_col).max()).item()
        if x_min is not None and x_max is not None:
            x_min = int(x_min)
            x_max = int(x_max)
            span = max(1, x_max - x_min + 1)
            bin_size = max(1, int(np.ceil(span / n_bins)))
            x_bin_col = f"{x_col}_bin"
            if bin_reducer != "mean":
                raise ValueError(f"Unsupported bin_reducer: {bin_reducer}")
            df = df.with_columns(
                (pl.col(x_col) // bin_size * bin_size).alias(x_bin_col))
            group_keys = [x_bin_col]
            if color_col in df.columns:
                group_keys = [color_col, *line_cols, x_bin_col]
            elif line_cols:
                group_keys = [*line_cols, x_bin_col]
            df = df.group_by(group_keys).agg(pl.col(y_col).mean().alias(y_col))
            x_col = x_bin_col

    if set_ylim_quantiles:
        qlo, qhi = ylim_quantiles
        y_all = df.select(pl.col(y_col)).to_series().to_numpy()
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size:
            a = float(np.quantile(y_all, qlo))
            b = float(np.quantile(y_all, qhi))
            d = b - a
            pad = 0.1 * (d if d > 0 else 1.0)
            ax.set_ylim(a - pad, b + pad)

    if yscale == "log":
        ax.set_yscale("log")
    elif yscale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh)
    else:
        ax.set_yscale("linear")

    groups = sorted(
        df[color_col].unique().to_list()) if color_col in df.columns else [
            "all"
        ]
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(groups))))

    for i, g in enumerate(groups):
        d0 = df.filter(
            pl.col(color_col) == g) if color_col in df.columns else df
        color = colors[i % len(colors)]
        if line_cols:
            first = True
            for _, d in d0.group_by(list(line_cols), maintain_order=True):
                d = d.sort(x_col)
                x = d[x_col].to_numpy()
                y = d[y_col].to_numpy()
                label = g if first else None
                ax.plot(x,
                        y,
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        label=label)
                first = False
        else:
            d = d0.sort(x_col)
            x = d[x_col].to_numpy()
            y = d[y_col].to_numpy()
            ax.plot(x,
                    y,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=g)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    if show_legend and groups and groups != ["all"]:
        ax.legend(loc="best")


def plot_actor_critic_dashboard(by_type: dict[str, pl.DataFrame],
                                *,
                                out: str = "dashboard.png"):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex="col")
    panels = [
        ("loss", ("critic_loss", "actor_loss"), "train_iter"),
        ("weight_norm", ("critic", "actor"), "train_iter"),
        ("grad_norm", ("critic", "actor"), "train_iter"),
    ]

    for r, (event_type, (critic_y, actor_y), x_col) in enumerate(panels):
        df = by_type.get(event_type, pl.DataFrame())
        if not df.is_empty() and x_col in df.columns:
            df = df.with_columns(pl.col(x_col).cast(pl.Int64))
        if not df.is_empty() and "agent_idx" in df.columns:
            df = df.with_columns(pl.col("agent_idx").cast(pl.Int64))

        ax_c = axes[r, 0]
        ax_a = axes[r, 1]
        plot_many_lines(ax_c,
                        df,
                        x_col=x_col,
                        y_col=critic_y,
                        title=f"{event_type}: critic",
                        n_bins=N_BINS if x_col == "train_iter" else None)
        plot_many_lines(ax_a,
                        df,
                        x_col=x_col,
                        y_col=actor_y,
                        title=f"{event_type}: actor",
                        n_bins=N_BINS if x_col == "train_iter" else None)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")


def plot_critic_dashboard(by_type: dict[str, pl.DataFrame],
                          *,
                          out: str = OUT_CRITIC):
    """Plot SAC critic-debug signals (no loss/grad/norm duplication)."""
    df = by_type.get("loss", pl.DataFrame())
    if df.is_empty():
        print("No loss records found; skipping critic dashboard.")
        return
    if "train_iter" in df.columns:
        df = df.with_columns(pl.col("train_iter").cast(pl.Int64))
    if "agent_idx" in df.columns:
        df = df.with_columns(pl.col("agent_idx").cast(pl.Int64))

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 12), sharex=True)

    # 1) log_alpha
    ax = axes[0]
    plot_many_lines(ax,
                    df,
                    x_col="train_iter",
                    y_col="log_alpha",
                    title="log_alpha",
                    n_bins=N_BINS,
                    set_ylim_quantiles=False)

    # Helper to overlay mean/min/max on same axis
    def _overlay_stats(ax,
                       *,
                       title: str,
                       mean_col: str,
                       min_col: str,
                       max_col: str,
                       yscale: Literal["linear", "log", "symlog"] = "linear"):
        if mean_col not in df.columns:
            ax.set_axis_off()
            return
        plot_many_lines(ax,
                        df,
                        x_col="train_iter",
                        y_col=mean_col,
                        title=title,
                        n_bins=N_BINS,
                        set_ylim_quantiles=False,
                        linestyle="-",
                        yscale=yscale,
                        alpha=0.35,
                        linewidth=1.2,
                        show_legend=True)
        if min_col in df.columns:
            plot_many_lines(ax,
                            df,
                            x_col="train_iter",
                            y_col=min_col,
                            n_bins=N_BINS,
                            set_ylim_quantiles=False,
                            linestyle="--",
                            yscale=yscale,
                            alpha=0.25,
                            linewidth=1.0,
                            show_legend=False)
        if max_col in df.columns:
            plot_many_lines(ax,
                            df,
                            x_col="train_iter",
                            y_col=max_col,
                            n_bins=N_BINS,
                            set_ylim_quantiles=False,
                            linestyle="--",
                            yscale=yscale,
                            alpha=0.25,
                            linewidth=1.0,
                            show_legend=False)

    # 2) log_pi stats
    _overlay_stats(axes[1],
                   title="log_pi (mean / min / max)",
                   mean_col="log_pi_mean",
                   min_col="log_pi_min",
                   max_col="log_pi_max",
                   yscale="linear")

    # 3) entropy reward stats (-alpha * log_pi), can spike / change sign
    _overlay_stats(axes[2],
                   title="entropy_reward = -alpha * log_pi (mean / min / max)",
                   mean_col="entropy_reward_mean",
                   min_col="entropy_reward_min",
                   max_col="entropy_reward_max",
                   yscale="symlog")

    # 4) target_q stats
    _overlay_stats(axes[3],
                   title="target_q (mean / min / max)",
                   mean_col="target_q_mean",
                   min_col="target_q_min",
                   max_col="target_q_max",
                   yscale="symlog")

    axes[-1].set_xlabel("train_iter")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")


def process_one_folder(*, name: str, folder: str, idx: int,
                       total: int) -> None:
    print(f"\n=== [{idx}/{total}] Processing {name} ({folder}) ===")
    out_dir = Path("plots") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type = load_by_type(folder=folder, names=NAMES)

    ep = by_type.get("episode", pl.DataFrame())
    if not ep.is_empty() and "episode_idx" in ep.columns:
        ep = ep.with_columns(pl.col("episode_idx").cast(pl.Int64))
    if not ep.is_empty() and "agent_idx" in ep.columns:
        ep = ep.with_columns(pl.col("agent_idx").cast(pl.Int64))

    out_iqm = str(out_dir / OUT)
    out_lines = str(out_dir / OUT_LINES)
    out_dash = str(out_dir / "dashboard.png")
    out_critic = str(out_dir / OUT_CRITIC)

    if ep.is_empty():
        print("No episode records found; skipping episode plots.")
    else:
        print("Generating episode plot dataframe...")
        plot_df = episode_plot_df(ep,
                                  max_episode=MAX_EPISODE,
                                  confidence=CONFIDENCE,
                                  n_boot=N_BOOT,
                                  bootstrap_seed=BOOTSTRAP_SEED,
                                  n_bins=N_BINS)
        print("Plotting episode IQM...")
        plot_episode_iqm(plot_df, out=out_iqm, confidence=CONFIDENCE)
        print(f"Saved plot to: {out_iqm}")

        print("Plotting episode returns (many-lines)...")
        fig, ax = plt.subplots(figsize=(10, 6))
        line_cols: tuple[str, ...] = (
            "seed", "agent_idx") if "agent_idx" in ep.columns else ("seed", )
        plot_many_lines(ax,
                        ep,
                        x_col="episode_idx",
                        y_col="episode_return",
                        line_cols=line_cols,
                        color_col="experiment",
                        title="Episode Return (many lines)",
                        n_bins=None)
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Episode Return")
        plt.tight_layout()
        plt.savefig(out_lines, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {out_lines}")

    plot_actor_critic_dashboard(by_type, out=out_dash)
    print(f"Saved plot to: {out_dash}")

    plot_critic_dashboard(by_type, out=out_critic)
    print(f"Saved plot to: {out_critic}")


if __name__ == "__main__":
    folders = _resolve_folders(FOLDER)
    if not folders:
        raise RuntimeError("No folders resolved from FOLDER")

    for i, (name, folder) in enumerate(folders, start=1):
        process_one_folder(name=name, folder=folder, idx=i, total=len(folders))
