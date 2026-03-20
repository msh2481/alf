#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plot_common import iqm, load_by_type, resolve_folders


DEFAULT_FOLDER = "/tmp/dmc/Rotator"
DEFAULT_NAMES = [
    "model_sac_grad_1",
    "model_sac_grad_1e1",
    "model_sac_grad_1e2",
    "model_sac_grad_1e3",
    "model_sac_grad_1e4",
]
LOSS_BIN_SIZE = 50


def _cmap_colors(name: str) -> np.ndarray:
    cmap = plt.get_cmap(name)
    return np.asarray(cmap.colors)


def _hide_ax(ax, title: str, reason: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, reason, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _prepare_event_df(df: pl.DataFrame, *, x_col: str) -> pl.DataFrame:
    if df.is_empty() or x_col not in df.columns:
        return df

    df = df.with_columns(pl.col(x_col).cast(pl.Int64))
    if "agent_idx" in df.columns:
        df = df.with_columns(pl.col("agent_idx").cast(pl.Int64))

    keys = ["experiment", "seed"]
    if "agent_idx" in df.columns:
        keys.append("agent_idx")
    keys.append(x_col)

    numeric_cols: list[str] = []
    first_cols: list[str] = []
    for c, dtype in df.schema.items():
        if c in keys:
            continue
        if dtype in (pl.Utf8, pl.String):
            first_cols.append(c)
        else:
            numeric_cols.append(c)

    aggs: list[pl.Expr] = []
    aggs.extend([pl.col(c).mean().alias(c) for c in numeric_cols])
    aggs.extend([pl.col(c).first().alias(c) for c in first_cols])
    return df.group_by(keys, maintain_order=True).agg(aggs).sort(keys)


def _bin_train_iter(df: pl.DataFrame, *, x_col: str = "train_iter",
                    bin_size: int = LOSS_BIN_SIZE) -> pl.DataFrame:
    if df.is_empty() or x_col not in df.columns or bin_size <= 1:
        return df
    b = int(bin_size)
    df = df.with_columns(
        ((pl.col(x_col).cast(pl.Float64) / b).floor() * b).cast(
            pl.Int64).alias(x_col))
    return _prepare_event_df(df, x_col=x_col)


def bootstrap_ci(
    df: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    x_col: str,
    value_col: str,
    stat: Literal["iqm", "mean"] = "iqm",
    confidence: float = 0.9,
    n_boot: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    if df.is_empty() or value_col not in df.columns or x_col not in df.columns:
        return pl.DataFrame()

    if stat == "iqm":
        stat_fn: Callable[[np.ndarray], float] = iqm
    elif stat == "mean":
        stat_fn = lambda a: float(np.mean(a)) if a.size else float("nan")
    else:
        raise ValueError(f"Unknown stat: {stat}")

    rng = np.random.default_rng(seed)
    keys = list(group_cols) + [x_col]
    out: list[dict[str, Any]] = []

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
        rec = {k: v for k, v in zip(keys, key_vals)}
        rec.update(
            stat=float(stat_fn(vals)),
            ci_low=float(np.percentile(boots, 100 * (alpha / 2))),
            ci_high=float(np.percentile(boots, 100 * (1 - alpha / 2))),
            n=int(n),
        )
        out.append(rec)
    return pl.DataFrame(out) if out else pl.DataFrame()


def episode_plot_df(
    ep: pl.DataFrame,
    *,
    confidence: float = 0.9,
    n_boot: int = 200,
    bootstrap_seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if ep.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    ep = _prepare_event_df(ep, x_col="episode_idx")
    if "episode_return" not in ep.columns:
        return pl.DataFrame(), pl.DataFrame()

    seed_curve = ep
    if "agent_idx" in ep.columns:
        seed_curve = ep.group_by(["experiment", "seed", "episode_idx"],
                                 maintain_order=True).agg(
                                     pl.col("episode_return").mean().alias(
                                         "episode_return")).sort(
                                             ["experiment", "seed",
                                              "episode_idx"])

    summary = bootstrap_ci(seed_curve,
                           group_cols=("experiment", ),
                           x_col="episode_idx",
                           value_col="episode_return",
                           stat="iqm",
                           confidence=confidence,
                           n_boot=n_boot,
                           seed=bootstrap_seed)
    return seed_curve, summary


def _plot_seed_lines_and_mean(ax,
                              df: pl.DataFrame,
                              *,
                              x_col: str,
                              y_col: str,
                              title: str,
                              color_col: str = "experiment",
                              line_cols: Sequence[str] = ("seed", "agent_idx"),
                              alpha: float = 0.2,
                              linewidth: float = 0.9,
                              mean_linewidth: float = 2.0,
                              yscale: Literal["linear", "log",
                                              "symlog"] = "linear",
                              symlog_linthresh: float = 1.0,
                              overlay_cols: Sequence[tuple[str, str,
                                                           str]] = ()) -> None:
    if df.is_empty() or x_col not in df.columns or y_col not in df.columns:
        _hide_ax(ax, title, f"No data for `{y_col}`")
        return

    line_cols = tuple(c for c in line_cols if c in df.columns)
    groups = sorted(df[color_col].unique().to_list())
    colors = _cmap_colors("Set1")

    if yscale == "log":
        ax.set_yscale("log")
    elif yscale == "symlog":
        ax.set_yscale("symlog", linthresh=symlog_linthresh)

    for i, group in enumerate(groups):
        color = colors[i % len(colors)]
        gdf = df.filter(pl.col(color_col) == group)

        if line_cols:
            first = True
            for _, sdf in gdf.group_by(list(line_cols), maintain_order=True):
                sdf = sdf.sort(x_col)
                ax.plot(sdf[x_col].to_numpy(),
                        sdf[y_col].to_numpy(),
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        label=group if first else None)
                first = False
        else:
            gdf = gdf.sort(x_col)
            ax.plot(gdf[x_col].to_numpy(),
                    gdf[y_col].to_numpy(),
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=group)

        mean_df = gdf.group_by([color_col, x_col], maintain_order=True).agg(
            pl.col(y_col).mean().alias(y_col)).sort(x_col)
        ax.plot(mean_df[x_col].to_numpy(),
                mean_df[y_col].to_numpy(),
                color=color,
                linewidth=mean_linewidth)

        for overlay_col, linestyle, suffix in overlay_cols:
            if overlay_col not in gdf.columns:
                continue
            overlay_df = gdf.group_by([color_col, x_col],
                                      maintain_order=True).agg(
                                          pl.col(overlay_col).mean().alias(
                                              overlay_col)).sort(x_col)
            ax.plot(overlay_df[x_col].to_numpy(),
                    overlay_df[overlay_col].to_numpy(),
                    color=color,
                    linewidth=1.2,
                    alpha=0.75,
                    linestyle=linestyle,
                    label=f"{group} {suffix}" if i == 0 else None)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _plot_episode_panel(ax, ep: pl.DataFrame, summary: pl.DataFrame) -> None:
    if ep.is_empty() or "episode_idx" not in ep.columns:
        _hide_ax(ax, "Episode Return", "No episode records found")
        return

    colors = _cmap_colors("Set1")
    groups = sorted(ep["experiment"].unique().to_list())
    for i, group in enumerate(groups):
        color = colors[i % len(colors)]
        gdf = ep.filter(pl.col("experiment") == group)
        first = True
        for _, sdf in gdf.group_by(["seed"], maintain_order=True):
            sdf = sdf.sort("episode_idx")
            ax.plot(sdf["episode_idx"].to_numpy(),
                    sdf["episode_return"].to_numpy(),
                    color=color,
                    alpha=0.2,
                    linewidth=0.9,
                    label=group if first else None)
            first = False

        if not summary.is_empty():
            s = summary.filter(pl.col("experiment") == group).sort("episode_idx")
            if not s.is_empty():
                x = s["episode_idx"].to_numpy()
                y = s["stat"].to_numpy()
                lo = s["ci_low"].to_numpy()
                hi = s["ci_high"].to_numpy()
                ax.plot(x, y, color=color, linewidth=2.0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.12)

    ax.set_title("Episode Return IQM")
    ax.set_xlabel("episode_idx")
    ax.set_ylabel("episode_return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_dashboard(by_type: dict[str, pl.DataFrame], *, out: str) -> None:
    ep = by_type.get("episode", pl.DataFrame())
    loss = by_type.get("loss", pl.DataFrame())

    ep_seed, ep_summary = episode_plot_df(ep)
    loss = _prepare_event_df(loss, x_col="train_iter")
    loss = _bin_train_iter(loss, x_col="train_iter", bin_size=LOSS_BIN_SIZE)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
    axes = axes.reshape(-1)

    _plot_episode_panel(axes[0], ep_seed, ep_summary)
    _plot_seed_lines_and_mean(axes[1],
                              loss,
                              x_col="train_iter",
                              y_col="model_loss_mean",
                              title="Model Loss",
                              yscale="symlog",
                              symlog_linthresh=1e-3)
    _plot_seed_lines_and_mean(axes[2],
                              loss,
                              x_col="train_iter",
                              y_col="grad_sync_loss_mean",
                              title="Grad Sync Loss",
                              yscale="symlog",
                              symlog_linthresh=1e-3,
                              overlay_cols=(("grad_sync_loss_min", "--", "min"),
                                            ("grad_sync_loss_max", "--",
                                             "max")))
    _plot_seed_lines_and_mean(axes[3],
                              loss,
                              x_col="train_iter",
                              y_col="grad_align_cos_mean",
                              title="Grad Alignment Cosine",
                              yscale="linear",
                              overlay_cols=(("grad_align_cos_min", "--", "min"),
                                            ("grad_align_cos_max", "--",
                                             "max")))
    _plot_seed_lines_and_mean(axes[4],
                              loss,
                              x_col="train_iter",
                              y_col="critic_dqda_norm_mean",
                              title="dQ/da Norms",
                              yscale="symlog",
                              symlog_linthresh=1e-4,
                              overlay_cols=(("model_dqda_norm_mean", "-.",
                                             "model"), ))
    _plot_seed_lines_and_mean(axes[5],
                              loss,
                              x_col="train_iter",
                              y_col="critic_loss",
                              title="Critic Loss",
                              yscale="symlog",
                              symlog_linthresh=1e-4)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_folder(*, name: str, folder: str, names: list[str],
                   out_root: Path) -> None:
    print(f"\n=== Processing {name} ({folder}) ===")
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type = load_by_type(folder, names)
    out_path = out_dir / "sac_grad_dashboard.png"
    plot_dashboard(by_type, out=str(out_path))
    print(f"Saved plot to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot combined SAC-Grad diagnostics dashboard.")
    parser.add_argument("--folder",
                        type=str,
                        default=DEFAULT_FOLDER,
                        help="Root folder containing experiment runs.")
    parser.add_argument("--names",
                        nargs="+",
                        default=DEFAULT_NAMES,
                        help="Experiment names to include.")
    parser.add_argument("--out_dir",
                        type=str,
                        default="plots_sac_grad",
                        help="Output directory for plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folders = resolve_folders(args.folder)
    if not folders:
        raise RuntimeError(f"No folders resolved from {args.folder}")

    out_root = Path(args.out_dir)
    for name, folder in folders:
        process_folder(name=name,
                       folder=folder,
                       names=list(args.names),
                       out_root=out_root)


if __name__ == "__main__":
    main()
