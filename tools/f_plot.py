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
"""Plot helper specialized for experiments named `a{num_agents}_f{fraction}`.

Compared to `custom_plot.py`, the IQM plot is formatted for many curves:
- Color encodes number of agents (and the CI band color follows).
- Linestyle encodes own-fraction.
- Two legends are shown: one for colors, one for linestyles.

Everything else (loading, binning, bootstrap IQM CI) matches `custom_plot.py`.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

from plot_common import (agent_reduce, iqm, load_by_type, resolve_folders as
                         _resolve_folders)

# --- User config (hardcoded-first) -------------------------------------------------
#
# FOLDER can be:
# - a single folder path (str)
# - a list/tuple of folder paths
# - the special string "all_dm", which expands to all subfolders of /tmp/dmc.
FOLDER: str | Sequence[str] = "/tmp/dmc/cheetah_run"

# If None, experiment names are auto-discovered from subfolders of each folder
# using the `a{num}_f{fraction}` pattern.
NAMES: list[str] | None = None

# Optional explicit (agents, fractions) to generate names when NAMES is None.
# If provided (non-empty), this takes priority over auto-discovery.
AGENTS: list[int] = []
FRACTIONS: list[float] = []

OUT_IQM_MEAN = "iqm_mean.png"
OUT_IQM_MAX = "iqm_max.png"
MAX_EPISODE: int | None = None
CONFIDENCE = 0.9
N_BOOT = 100
BOOTSTRAP_SEED = 0
PLOT_RETURN_QUANTILES = False
IQM_CI_ALPHA = 0.05

# Rendering controls to keep dense plots readable (and zoomable).
FIGSIZE = (14, 8)
LINEWIDTH_MIN = 0.75
LINEWIDTH_MAX = 1.7
LINE_ALPHA_MAX = 1.0
LINE_ALPHA_MIN = 0.35
SAVE_DPI = 600
USE_LINE_HALO = False
HALO_COLOR = "white"
HALO_WIDTH = 2.0  # additional width under the colored line

# Only consider these `f` values (set to None to disable filtering).
ALLOWED_FRACTIONS: list[float] | None = [
    1 / 32,
    1 / 16,
    1 / 8,
    1 / 4,
    1 / 2,
    3 / 5,
    3 / 4,
]

# Episode index correction:
# In our logs, `episode_idx` is tracked per-agent. If each agent controls fewer
# environments when num_agents is larger, per-agent episode counts are not
# directly comparable across num_agents. When enabled, we scale the plotted
# x-coordinate by (num_agents / EPISODE_INDEX_BASE_AGENTS) so that the x-axis
# approximates "iterations of the whole concurrent setup".
CORRECT_EPISODES = True
EPISODE_INDEX_BASE_AGENTS = 32

# In dense grids of (agents x fractions) curves, jitter is more confusing than
# helpful. Keep it off by default.
IQM_LINE_JITTER_FRAC = 0.0

BIN_CONF: dict[str, tuple[str, int]] = {
    "episode": ("episode_idx", 10),
    "loss": ("train_iter", 1000),
    "weight_norm": ("train_iter", 1000),
    "grad_norm": ("train_iter", 1000),
}

# `experiment` naming scheme we target.
_AF_RE = re.compile(r"^a(?P<a>\d+)_f(?P<f>[0-9]*\.?[0-9]+)$")


def _parse_a_f(name: str) -> tuple[int | None, float | None]:
    m = _AF_RE.match(name.strip())
    if not m:
        return None, None
    try:
        a = int(m.group("a"))
        f = float(m.group("f"))
        return a, f
    except Exception:
        return None, None


def _format_fraction(f: float) -> str:
    """Human-friendly fraction label (e.g. 0.125 -> 1/8, 0.6 -> 0.60)."""
    if not np.isfinite(f):
        return str(f)
    fr = Fraction(float(f)).limit_denominator(64)
    approx = fr.numerator / fr.denominator
    if abs(approx - float(f)) <= 1e-9:
        return f"{fr.numerator}/{fr.denominator}"
    # Fall back to a short decimal.
    return f"{float(f):.2f}".rstrip("0").rstrip(".")


def _discover_af_names(folder: str) -> list[str]:
    root = Path(folder).expanduser()
    if not root.exists():
        return []
    names: list[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if _AF_RE.match(p.name):
            names.append(p.name)
    # Stable sort by (agents, fraction).
    def _key(n: str) -> tuple[int, float, str]:
        a, f = _parse_a_f(n)
        if a is None or f is None:
            return (10**9, float("inf"), n)
        return (a, f, n)

    return sorted(names, key=_key)


def _names_for_folder(folder: str) -> list[str]:
    if NAMES is not None:
        names = list(NAMES)
    elif AGENTS and FRACTIONS:
        # Keep stable grouping: agent-major, then fraction.
        names = [f"a{int(a)}_f{float(f)}" for a in AGENTS for f in FRACTIONS]
    else:
        names = _discover_af_names(folder)

    if not names or ALLOWED_FRACTIONS is None:
        return names

    allowed = [float(f) for f in ALLOWED_FRACTIONS]
    tol = 1e-9

    def _ok(name: str) -> bool:
        _a, f = _parse_a_f(name)
        if f is None or not np.isfinite(f):
            return False
        return any(abs(float(f) - af) <= tol for af in allowed)

    return [n for n in names if _ok(n)]


def _episode_x_scale(num_agents: int) -> float:
    base = float(EPISODE_INDEX_BASE_AGENTS)
    if not np.isfinite(base) or base <= 0:
        return 1.0
    return float(num_agents) / base


def _xlabel_episode() -> str:
    return "Episode Index (x32)" if CORRECT_EPISODES else "Episode Index"


def _bin_and_reduce(df: pl.DataFrame, *, x_col: str,
                    bin_size: int) -> pl.DataFrame:
    """Bin the x-axis in-place and reduce duplicates within each curve.

    - Overwrites `x_col` with its binned value (no `*_bin` column).
    - Reduces within (experiment, seed, [agent_idx], x_col) by taking mean of
      numeric columns and first of non-numeric columns.
    """
    if df.is_empty() or x_col not in df.columns:
        return df

    df = df.with_columns(pl.col(x_col).cast(pl.Int64))
    if bin_size > 1:
        df = df.with_columns(
            (pl.col(x_col) // int(bin_size) * int(bin_size)).alias(x_col))

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
    # Most qualitative maps expose `.colors` as a list. Fall back to sampling.
    colors = getattr(cmap, "colors", None)
    if colors is not None:
        return np.asarray(colors)
    return cmap(np.linspace(0, 1, 10))


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


def plot_episode_iqm_af(plot_df: pl.DataFrame,
                        *,
                        out: str,
                        group_col: str = "experiment",
                        confidence: float = 0.9):
    """IQM plot for `a{num_agents}_f{fraction}` with dual legends."""
    if plot_df.is_empty() or group_col not in plot_df.columns:
        print(
            f"Empty plot_df or missing '{group_col}'; skipping IQM plot: {out}"
        )
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    groups = sorted(plot_df[group_col].unique().to_list())

    parsed = {g: _parse_a_f(g) for g in groups}
    agents = sorted({a for a, f in parsed.values() if a is not None})
    fracs = sorted({f for a, f in parsed.values() if f is not None})

    # Fallback: if names don't match, behave like the original (single legend).
    if not agents or not fracs:
        colors = _cmap_colors("Set1")
        for i, g in enumerate(groups):
            x_col = "episode_idx"
            d = plot_df.filter(pl.col(group_col) == g).sort(x_col)
            x = d[x_col].to_numpy()
            y = d["stat"].to_numpy()
            lo = d["ci_low"].to_numpy()
            hi = d["ci_high"].to_numpy()
            q25 = d["q25"].to_numpy()
            q75 = d["q75"].to_numpy()
            color = colors[i % len(colors)]
            ax.plot(x, y, label=g, color=color, linewidth=1)
            ax.fill_between(x, lo, hi, alpha=IQM_CI_ALPHA, color=color)
            if PLOT_RETURN_QUANTILES:
                ax.plot(x, q25, "--", color=color, linewidth=1, alpha=0.5)
                ax.plot(x, q75, "--", color=color, linewidth=1, alpha=0.5)
        ax.set_xlabel(_xlabel_episode())
        ax.set_ylabel("Episode Return")
        ax.set_title(f"IQM Episode Return with {int(confidence * 100)}% CI")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
        return

    colors = _cmap_colors("tab10")
    agent_to_color = {a: colors[i % len(colors)] for i, a in enumerate(agents)}

    # Linestyle mapping (increasing f -> more "solid"):
    # - <= 1/8: dotted
    # - 1/4: dash-dot-ish ("-.-" intent)
    # - then gradually: dashed -> long-dash -> solid.
    #
    # Matplotlib doesn't have a "-.-" token; this custom dash pattern renders
    # close to that intent.
    ls_dashdotish = (0, (6, 2, 1.5, 2))  # dash, gap, dot, gap
    frac_to_ls: dict[float, Any] = {}
    for f in fracs:
        ff = float(f)
        if ff <= float(1 / 8) + 1e-12:
            frac_to_ls[f] = ":"
        else:
            # For typical allowed set, this yields:
            # 1/4 dash-dot-ish, 1/2 '--', 3/5 long-dash, 3/4 '-'.
            if abs(ff - float(1 / 4)) <= 1e-9:
                frac_to_ls[f] = ls_dashdotish
            elif abs(ff - float(1 / 2)) <= 1e-9:
                frac_to_ls[f] = "--"
            elif abs(ff - float(3 / 5)) <= 1e-9:
                frac_to_ls[f] = (0, (10, 2))  # long-dash (closer to solid than '--')
            elif abs(ff - float(3 / 4)) <= 1e-9:
                frac_to_ls[f] = "-"
            else:
                # Rank-based fallback: dash-dot-ish -> dashed -> long-dash -> solid.
                bigger = sorted([x for x in fracs if float(x) > float(1 / 8) + 1e-12])
                idx = bigger.index(f) if f in bigger else 0
                ladder: list[Any] = [ls_dashdotish, "--", (0, (10, 2)), "-"]
                frac_to_ls[f] = ladder[min(idx, len(ladder) - 1)]

    f_min = float(np.min(fracs)) if fracs else 0.0
    f_max = float(np.max(fracs)) if fracs else 1.0
    f_rng = max(f_max - f_min, 1e-12)

    def _lw_for_f(ff: float) -> float:
        t = (float(ff) - f_min) / f_rng
        return float(LINEWIDTH_MIN + t * (LINEWIDTH_MAX - LINEWIDTH_MIN))

    def _alpha_for_f(ff: float) -> float:
        # Higher f -> slightly more transparent (but keep within a narrow band).
        t = (float(ff) - f_min) / f_rng
        return float(LINE_ALPHA_MAX - t * (LINE_ALPHA_MAX - LINE_ALPHA_MIN))

    x_col = "episode_idx"
    for i, g in enumerate(groups):
        a, f = parsed.get(g, (None, None))
        if a is None or f is None:
            continue
        d = plot_df.filter(pl.col(group_col) == g).sort(x_col)
        x = d[x_col].to_numpy()
        if CORRECT_EPISODES:
            x = x.astype(np.float64, copy=False) * _episode_x_scale(int(a))
        y = d["stat"].to_numpy()
        lo = d["ci_low"].to_numpy()
        hi = d["ci_high"].to_numpy()
        q25 = d["q25"].to_numpy()
        q75 = d["q75"].to_numpy()

        color = agent_to_color[a]
        ls = frac_to_ls[f]
        lw = _lw_for_f(float(f))
        la = _alpha_for_f(float(f))

        if IQM_LINE_JITTER_FRAC and len(groups) > 1:
            y_scale = float(np.nanmax(np.abs(y))) if y.size else 1.0
            y_scale = max(y_scale, 1.0)
            offset = (i - (len(groups) - 1) / 2.0) * float(
                IQM_LINE_JITTER_FRAC) * y_scale
        else:
            offset = 0.0

        line = ax.plot(x,
                       y + offset,
                       color=color,
                       linestyle=ls,
                       linewidth=lw,
                       alpha=la,
                       zorder=3)[0]
        if USE_LINE_HALO:
            line.set_path_effects([
                pe.Stroke(linewidth=lw + HALO_WIDTH, foreground=HALO_COLOR),
                pe.Normal(),
            ])
        ax.fill_between(x,
                        lo + offset,
                        hi + offset,
                        alpha=IQM_CI_ALPHA,
                        color=color,
                        zorder=1)
        if PLOT_RETURN_QUANTILES:
            ax.plot(x,
                    q25 + offset,
                    linestyle=ls,
                    color=color,
                    linewidth=max(0.6, 0.7 * lw),
                    alpha=0.5 * la,
                    zorder=2)
            ax.plot(x,
                    q75 + offset,
                    linestyle=ls,
                    color=color,
                    linewidth=max(0.6, 0.7 * lw),
                    alpha=0.5 * la,
                    zorder=2)

    # Dual legends (color = agents, linestyle = fraction).
    color_handles = [
        Line2D([0], [0],
               color=agent_to_color[a],
               linestyle="-",
               linewidth=2.0,
               label=f"a={a}") for a in agents
    ]
    frac_handles = [
        Line2D([0], [0],
               color="black",
               linestyle=frac_to_ls[f],
               linewidth=_lw_for_f(float(f)),
               alpha=_alpha_for_f(float(f)),
               label=f"f={_format_fraction(float(f))}") for f in fracs
    ]

    leg1 = ax.legend(handles=color_handles,
                     title="num agents",
                     loc="upper left",
                     frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=frac_handles,
              title="own fraction",
              loc="upper right",
              frameon=True)

    ax.set_xlabel(_xlabel_episode())
    ax.set_ylabel("Episode Return")
    ax.set_title(f"IQM Episode Return with {int(confidence * 100)}% CI")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")


def process_one_folder(*, name: str, folder: str, idx: int,
                       total: int) -> None:
    print(f"\n=== [{idx}/{total}] Processing {name} ({folder}) ===")
    out_dir = Path("plots") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    names = _names_for_folder(folder)
    if not names:
        print(f"No experiments found for pattern `a{{N}}_f{{F}}` in {folder}; skipping.")
        return

    by_type = load_by_type(folder=folder, names=names)
    by_type = _preprocess_by_type(by_type, max_episode=MAX_EPISODE)

    ep = by_type.get("episode", pl.DataFrame())

    out_iqm_mean = str(out_dir / OUT_IQM_MEAN)
    out_iqm_max = str(out_dir / OUT_IQM_MAX)

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
        plot_episode_iqm_af(plot_df_mean, out=out_iqm_mean, confidence=CONFIDENCE)
        print(f"Saved plot to: {out_iqm_mean}")

        print("Generating episode plot dataframe (agent-max)...")
        plot_df_max = episode_plot_df(ep,
                                      max_episode=MAX_EPISODE,
                                      confidence=CONFIDENCE,
                                      n_boot=N_BOOT,
                                      bootstrap_seed=BOOTSTRAP_SEED,
                                      agent_reduce="max")
        print("Plotting episode IQM (agent-max)...")
        plot_episode_iqm_af(plot_df_max, out=out_iqm_max, confidence=CONFIDENCE)
        print(f"Saved plot to: {out_iqm_max}")


if __name__ == "__main__":
    folders = _resolve_folders(FOLDER)
    if not folders:
        raise RuntimeError("No folders resolved from FOLDER")

    for i, (name, folder) in enumerate(folders, start=1):
        process_one_folder(name=name, folder=folder, idx=i, total=len(folders))

