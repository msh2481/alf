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

import argparse
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from plot_common import (agent_reduce, iqm, load_by_type, resolve_folders as
                         _resolve_folders)

# FOLDER = "/tmp/dmc/Rotator"
# FOLDER = "/tmp/dmc/pendulum_swingup"
# FOLDER can be:
# - a single folder path (str)
# - a list/tuple of folder paths
# - the special string "all_dm", which expands to all
#   subfolders of /tmp/dmc.
FOLDER: str | Sequence[str] = "all_dm"
# FOLDER: str | Sequence[str] = "/tmp/bipolar/BipolarChain-medium-sparse-onehot-discrete-v0/"
NAMES = [
    "a4_prior0",
    "a4_prior0.001",
    "a4_prior0.01",
    "a4_prior0.1",
    "a4_prior1.0",
] 

OUT_IQM_MEAN = "iqm_mean.png"
OUT_IQM_MAX = "iqm_max.png"
OUT_LINES = "lines_episode_return.png"
OUT_CRITIC = "critic.png"
MAX_EPISODE: int | None = None
CONFIDENCE = 0.9
N_BOOT = 100
BOOTSTRAP_SEED = 0
PLOT_RETURN_QUANTILES = False
IQM_CI_ALPHA = 0.1
IQM_LINE_JITTER_FRAC = 5e-3

# Episode index correction:
# In our logs, `episode_idx` is tracked per-agent. If each agent controls fewer
# environments when num_agents is larger, per-agent episode counts are not
# directly comparable across num_agents. When enabled, we scale the plotted
# x-coordinate by (num_agents / EPISODE_INDEX_BASE_AGENTS) so that the x-axis
# approximates "iterations of the whole concurrent setup".
CORRECT_EPISODES = True
EPISODE_INDEX_BASE_AGENTS = 32

BIN_CONF: dict[str, tuple[str, int]] = {
    "episode": ("episode_idx", 20),
    "loss": ("train_iter", 1000),
    "weight_norm": ("train_iter", 1000),
    "grad_norm": ("train_iter", 1000),
}

_A_PREFIX_RE = re.compile(r"^a(?P<a>\d+)(?:_|$)")


def _parse_num_agents(experiment_name: str) -> int | None:
    """Parse `a{num_agents}_...` style experiment names."""
    m = _A_PREFIX_RE.match(str(experiment_name).strip())
    if not m:
        return None
    try:
        return int(m.group("a"))
    except Exception:
        return None


def _episode_x_scale(num_agents: int) -> float:
    base = float(EPISODE_INDEX_BASE_AGENTS)
    if not np.isfinite(base) or base <= 0:
        return 1.0
    return float(num_agents) / base


def _xlabel_episode() -> str:
    return f"Episode Index (x{EPISODE_INDEX_BASE_AGENTS})" if CORRECT_EPISODES else "Episode Index"


def _bin_and_reduce(df: pl.DataFrame, *, x_col: str,
                    bin_size: int) -> pl.DataFrame:
    """Bin the x-axis in-place and reduce duplicates within each curve.

    - Overwrites `x_col` with its binned value (no `*_bin` column).
    - Reduces within (experiment, seed, [agent_idx], x_col) by taking mean of
      numeric columns and first of non-numeric columns.
    """
    if df.is_empty() or x_col not in df.columns:
        return df

    # Ensure integer x-axis for stable binning and plotting.
    df = df.with_columns(pl.col(x_col).cast(pl.Int64))

    if bin_size > 1:
        df = df.with_columns(
            (pl.col(x_col) // int(bin_size) * int(bin_size)).alias(x_col))

    # Reduce duplicates / within-bin points so each (seed[,agent],x) is a single sample.
    group_keys: list[str] = ["experiment", "seed"]
    if "agent_idx" in df.columns:
        group_keys.append("agent_idx")
    group_keys.append(x_col)

    numeric_cols: list[str] = []
    first_cols: list[str] = []
    for c, dtype in df.schema.items():
        if c in group_keys:
            continue
        if dtype in (pl.Utf8, pl.String):
            first_cols.append(c)
        else:
            numeric_cols.append(c)

    aggs: list[pl.Expr] = []
    aggs.extend([pl.col(c).mean().alias(c) for c in numeric_cols])
    aggs.extend([pl.col(c).first().alias(c) for c in first_cols])

    out = df.group_by(group_keys, maintain_order=True).agg(aggs)
    return out.sort(group_keys)


def _preprocess_by_type(by_type: dict[str, pl.DataFrame], *,
                        max_episode: int | None) -> dict[str, pl.DataFrame]:
    """Apply BIN_CONF (and episode max filter) immediately after loading."""
    out: dict[str, pl.DataFrame] = {}
    for event_type, df in by_type.items():
        if df.is_empty():
            out[event_type] = df
            continue

        conf = BIN_CONF.get(event_type)
        if conf is None:
            out[event_type] = df
            continue

        x_col, bin_size = conf

        if event_type == "episode" and max_episode is not None and "episode_idx" in df.columns:
            df = df.filter(pl.col("episode_idx") <= int(max_episode))

        out[event_type] = _bin_and_reduce(df,
                                          x_col=x_col,
                                          bin_size=int(bin_size))
    return out


def bootstrap_ci(
    df: pl.DataFrame,
    *,
    group_cols: Sequence[str] = ("experiment", ),
    x_col: str,
    value_col: str,
    stat: Literal["iqm", "mean", "median"] = "iqm",
    confidence: float = 0.9,
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


def _cmap_colors(name: str) -> np.ndarray:
    cmap = plt.get_cmap(name)
    return np.asarray(cmap.colors)


def episode_plot_df(
        ep: pl.DataFrame,
        *,
        group_cols: Sequence[str] = ("experiment", ),
        max_episode: int | None = None,
        confidence: float = 0.9,
        n_boot: int = 2000,
        bootstrap_seed: int = 0,
        agent_reduce: Literal["mean", "max", "none"] = "mean") -> pl.DataFrame:
    if ep.is_empty():
        return pl.DataFrame()
    if max_episode is not None:
        ep = ep.filter(pl.col("episode_idx") <= max_episode)
    if not ep.is_empty() and "episode_idx" in ep.columns:
        ep = ep.with_columns(pl.col("episode_idx").cast(pl.Int64))
    x_col = "episode_idx"

    # If there are multiple agents per seed, reduce across agents first so they
    # are not treated as independent bootstrap samples by default.
    ep_reduced = ep
    if agent_reduce != "none" and "agent_idx" in ep.columns:
        keys = [*group_cols, "seed", x_col]
        if agent_reduce == "mean":
            agg = pl.col("episode_return").mean().alias("episode_return")
        elif agent_reduce == "max":
            agg = pl.col("episode_return").max().alias("episode_return")
        else:
            raise ValueError(f"Unknown agent_reduce: {agent_reduce}")
        ep_reduced = ep.group_by(keys, maintain_order=True).agg(agg).sort(keys)

    ci = bootstrap_ci(ep_reduced,
                      group_cols=group_cols,
                      x_col=x_col,
                      value_col="episode_return",
                      stat="iqm",
                      confidence=confidence,
                      n_boot=n_boot,
                      seed=bootstrap_seed)
    q = ep_reduced.group_by([*group_cols, x_col]).agg(
        q25=pl.col("episode_return").quantile(0.25),
        q75=pl.col("episode_return").quantile(0.75),
    )
    return ci.join(q, on=[*group_cols, x_col],
                   how="left").sort([*group_cols, x_col])


def plot_episode_iqm(plot_df: pl.DataFrame,
                     *,
                     out: str,
                     group_col: str = "experiment",
                     confidence: float = 0.9):
    if plot_df.is_empty() or group_col not in plot_df.columns:
        print(
            f"Empty plot_df or missing '{group_col}'; skipping IQM plot: {out}"
        )
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(plot_df[group_col].unique().to_list())
    colors = _cmap_colors("Set1")
    n_groups = len(groups)
    scaled_any = False
    for i, g in enumerate(groups):
        x_col = "episode_idx"
        d = plot_df.filter(pl.col(group_col) == g).sort(x_col)
        x = d[x_col].to_numpy()
        if CORRECT_EPISODES:
            a = _parse_num_agents(str(g))
            if a is not None:
                x = x.astype(np.float64, copy=False) * _episode_x_scale(int(a))
                scaled_any = True
        y = d["stat"].to_numpy()
        lo = d["ci_low"].to_numpy()
        hi = d["ci_high"].to_numpy()
        q25 = d["q25"].to_numpy()
        q75 = d["q75"].to_numpy()
        color = colors[i % len(colors)]

        # Deterministic micro-jitter to separate identical curves.
        if IQM_LINE_JITTER_FRAC and n_groups > 1:
            y_scale = float(np.nanmax(np.abs(y))) if y.size else 1.0
            y_scale = max(y_scale, 1.0)
            offset = (i - (n_groups - 1) / 2.0) * float(
                IQM_LINE_JITTER_FRAC) * y_scale
        else:
            offset = 0.0

        ax.plot(x, y + offset, label=g, color=color, linewidth=1)
        ax.fill_between(x, lo + offset, hi + offset, alpha=IQM_CI_ALPHA, color=color)
        if PLOT_RETURN_QUANTILES:
            ax.plot(x, q25 + offset, "--", color=color, linewidth=1, alpha=0.5)
            ax.plot(x, q75 + offset, "--", color=color, linewidth=1, alpha=0.5)
    ax.set_xlabel(_xlabel_episode() if (CORRECT_EPISODES and scaled_any) else "Episode Index")
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
    colors = _cmap_colors("Set1")

    scaled_any = False
    for i, g in enumerate(groups):
        d0 = df.filter(
            pl.col(color_col) == g) if color_col in df.columns else df
        color = colors[i % len(colors)]
        x_scale = 1.0
        if x_col == "episode_idx" and CORRECT_EPISODES and color_col in df.columns:
            a = _parse_num_agents(str(g))
            if a is not None:
                x_scale = _episode_x_scale(int(a))
        if line_cols:
            first = True
            for _, d in d0.group_by(list(line_cols), maintain_order=True):
                d = d.sort(x_col)
                x = d[x_col].to_numpy()
                if x_scale != 1.0:
                    x = x.astype(np.float64, copy=False) * float(x_scale)
                    scaled_any = True
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
            if x_scale != 1.0:
                x = x.astype(np.float64, copy=False) * float(x_scale)
                scaled_any = True
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
    if x_col == "episode_idx" and CORRECT_EPISODES and scaled_any:
        ax.set_xlabel(_xlabel_episode())
    else:
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
                        title=f"{event_type}: critic")
        plot_many_lines(ax_a,
                        df,
                        x_col=x_col,
                        y_col=actor_y,
                        title=f"{event_type}: actor")

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


def process_one_folder(*, name: str, folder: str, idx: int, total: int,
                       names: Sequence[str]) -> None:
    print(f"\n=== [{idx}/{total}] Processing {name} ({folder}) ===")
    out_dir = Path("plots") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type = load_by_type(folder=folder, names=names)
    by_type = _preprocess_by_type(by_type, max_episode=MAX_EPISODE)

    ep = by_type.get("episode", pl.DataFrame())

    out_iqm_mean = str(out_dir / OUT_IQM_MEAN)
    out_iqm_max = str(out_dir / OUT_IQM_MAX)
    out_lines = str(out_dir / OUT_LINES)
    out_dash = str(out_dir / "dashboard.png")
    out_critic = str(out_dir / OUT_CRITIC)

    if ep.is_empty():
        print("No episode records found; skipping episode plots.")
    else:
        print("Generating episode plot dataframe (agent-mean)...")
        plot_df_mean = episode_plot_df(ep,
                                       max_episode=MAX_EPISODE,
                                       confidence=CONFIDENCE,
                                       n_boot=N_BOOT,
                                       bootstrap_seed=BOOTSTRAP_SEED,
                                       agent_reduce="mean")
        print("Plotting episode IQM (agent-mean)...")
        plot_episode_iqm(plot_df_mean, out=out_iqm_mean, confidence=CONFIDENCE)
        print(f"Saved plot to: {out_iqm_mean}")

        print("Generating episode plot dataframe (agent-max)...")
        plot_df_max = episode_plot_df(ep,
                                      max_episode=MAX_EPISODE,
                                      confidence=CONFIDENCE,
                                      n_boot=N_BOOT,
                                      bootstrap_seed=BOOTSTRAP_SEED,
                                      agent_reduce="max")
        print("Plotting episode IQM (agent-max)...")
        plot_episode_iqm(plot_df_max, out=out_iqm_max, confidence=CONFIDENCE)
        print(f"Saved plot to: {out_iqm_max}")

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
                        title="Episode Return (many lines)")
        plt.tight_layout()
        plt.savefig(out_lines, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {out_lines}")

    plot_actor_critic_dashboard(by_type, out=out_dash)
    print(f"Saved plot to: {out_dash}")

    plot_critic_dashboard(by_type, out=out_critic)
    print(f"Saved plot to: {out_critic}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate custom plots from experiment logs.")
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
    args = parser.parse_args()
    names = args.names or NAMES
    EPISODE_INDEX_BASE_AGENTS = int(args.episode_index_base_agents)

    folder = args.folder or FOLDER
    folders = _resolve_folders(folder)
    if not folders:
        raise RuntimeError("No folders resolved from FOLDER")

    for i, (name, folder) in enumerate(folders, start=1):
        process_one_folder(name=name,
                           folder=folder,
                           idx=i,
                           total=len(folders),
                           names=names)
