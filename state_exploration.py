#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
import polars as pl


def _load_table(path):
    return pl.read_ndjson(path)


def _rows(table):
    return table.sort(["rollout_step", "alg_idx"]).to_dicts()


def _to_np(obs):
    arr = np.asarray(obs, dtype=np.float64)
    return arr.reshape(-1) if arr.ndim > 1 else arr


def _flag_episode_end(value):
    if value is None:
        return False
    arr = np.asarray(value)
    return bool(arr.any())


class StateStore:

    def __init__(self, rows, k=None):
        self.rows = rows
        self.k = int(k) if k else None
        self.states = {}
        self.positions = {}
        self.events = []
        self.total = 0
        self._build()

    def _build(self):
        states = {}
        positions = {}
        counts = {}
        global_step = 0
        for row in self.rows:
            alg = int(row["alg_idx"])
            count = counts.get(alg, 0)
            if self.k is not None and count >= self.k:
                continue
            obs = _to_np(row["observation"])
            states.setdefault(alg, []).append(obs)
            global_step += 1
            positions.setdefault(alg, []).append(global_step)
            counts[alg] = count + 1
            local_idx = counts[alg] - 1
            self.events.append((global_step, alg, local_idx))
        self.states = {k: np.stack(v) for k, v in states.items() if v}
        self.positions = {
            k: np.array(v, dtype=np.int64)
            for k, v in positions.items() if v
        }
        self.total = global_step


def _encode_bins(arr, eps):
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    scaled = np.round(arr / eps).astype(np.int64)
    flat = np.ascontiguousarray(scaled)
    dtype = np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
    return flat.view(dtype).reshape(-1)


def _assemble_bins(store, eps):
    pieces = []
    offsets = {}
    offset = 0
    for alg in sorted(store.states.keys()):
        arr = store.states[alg]
        length = len(arr)
        offsets[alg] = (offset, offset + length)
        pieces.append(arr)
        offset += length
    if not pieces:
        return {}, np.zeros(0, dtype=np.uint8), {}
    stacked = np.concatenate(pieces, axis=0)
    bins_all = _encode_bins(stacked, eps)
    bins_per_alg = {}
    for alg, (lo, hi) in offsets.items():
        bins_per_alg[alg] = bins_all[lo:hi]
    return bins_per_alg, bins_all, offsets


def _new_state_steps(bins, positions):
    if len(bins) == 0:
        return np.zeros(0, dtype=np.int64)
    _, first_idx = np.unique(bins, return_index=True)
    steps = positions[first_idx]
    return np.sort(steps)


def _global_new_steps(bins_all, events, offsets):
    if len(bins_all) == 0:
        return np.zeros(0, dtype=np.int64)
    indices = []
    for _, alg, local_idx in events:
        start = offsets[alg][0]
        indices.append(start + local_idx)
    seq = bins_all[indices]
    _, first_idx = np.unique(seq, return_index=True)
    return np.sort(first_idx + 1)


def _counts(prefixes, steps):
    if len(steps) == 0:
        return np.zeros_like(prefixes)
    return np.searchsorted(steps, prefixes, side="right")


def _compute_feature_bounds(store, xs):
    if not store.states:
        return {}
    first = next(iter(store.states.values()))
    if first.size == 0:
        return {}
    dims = first.shape[1]
    bounds = {d: {} for d in range(dims)}
    for alg, arr in store.states.items():
        if arr.size == 0:
            continue
        cum_min = np.minimum.accumulate(arr, axis=0)
        cum_max = np.maximum.accumulate(arr, axis=0)
        pos = store.positions.get(alg)
        if pos is None or len(pos) == 0:
            continue
        counts = np.searchsorted(pos, xs, side="right")
        valid = counts > 0
        mins = np.full((len(xs), dims), np.nan)
        maxs = np.full((len(xs), dims), np.nan)
        if np.any(valid):
            idxs = counts[valid] - 1
            mins[valid] = cum_min[idxs]
            maxs[valid] = cum_max[idxs]
            for d in range(dims):
                bounds[d][alg] = (mins[:, d], maxs[:, d])
    return bounds


def _plot(xs, per_alg, overall, feature_bounds, outfile):
    dims = len(feature_bounds)
    cols = max(1, dims) if dims > 0 else 1
    if dims > 0:
        fig = plt.figure(figsize=(max(10, 4 * cols), 8))
        gs = fig.add_gridspec(2, cols, height_ratios=[1, 1])
        ax_top = fig.add_subplot(gs[0, :])
        bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(cols)]
    else:
        fig, ax_top = plt.subplots(figsize=(10, 6))
        bottom_axes = []
    colors = plt.cm.tab10.colors
    for i, alg in enumerate(sorted(per_alg.keys())):
        ax_top.plot(xs,
                    per_alg[alg],
                    label=f"alg {alg}",
                    linewidth=2,
                    color=colors[i % len(colors)])
    ax_top.plot(xs, overall, label="all", color="black", linestyle="--")
    ax_top.set_xlabel("global steps")
    ax_top.set_ylabel("unique states")
    ax_top.set_title("State exploration by prefix")
    ax_top.grid(True, alpha=0.2)
    ax_top.legend()
    for dim_idx, ax in enumerate(bottom_axes):
        bounds = feature_bounds.get(dim_idx, {})
        if not bounds:
            ax.axis("off")
            continue
        for i, alg in enumerate(sorted(bounds.keys())):
            color = colors[i % len(colors)]
            mins, maxs = bounds[alg]
            ax.plot(xs,
                    mins,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"alg {alg} min" if dim_idx == 0 else None)
            ax.plot(xs,
                    maxs,
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    label=f"alg {alg} max" if dim_idx == 0 else None)
        ax.set_title(f"dim {dim_idx}")
        ax.set_xlabel("global steps")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.2)
        if dim_idx == 0 and bounds:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"saved {outfile}")
    os.system(f"open '{outfile}'")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--points", type=int, default=200)
    p.add_argument("--outfile", type=str, default=None)
    a = p.parse_args()
    ndjson = os.path.join(a.log_dir, "rollout_states.ndjson")
    if not os.path.isfile(ndjson):
        print(f"missing {ndjson}", file=sys.stderr)
        sys.exit(1)
    rows = _rows(_load_table(ndjson))
    if not rows:
        print("no data", file=sys.stderr)
        sys.exit(1)
    store = StateStore(rows, k=a.k)
    if store.total == 0:
        print("no usable states", file=sys.stderr)
        sys.exit(1)
    bins_per_alg, bins_all, offsets = _assemble_bins(store, a.eps)
    per_alg_steps = {
        alg:
            _new_state_steps(
                bins_per_alg.get(alg, np.zeros(0, dtype=np.uint8)),
                store.positions.get(alg, np.zeros(0, dtype=np.int64)))
        for alg in store.states.keys()
    }
    global_steps = _global_new_steps(bins_all, store.events, offsets)
    points = store.total if a.points is None or a.points <= 0 else min(
        store.total, a.points)
    xs = np.linspace(1, store.total, points, dtype=np.int64)
    xs = np.unique(xs)
    if xs[-1] != store.total:
        xs = np.append(xs, store.total)
    per_alg_counts = {
        alg: _counts(xs, steps)
        for alg, steps in per_alg_steps.items()
    }
    overall_counts = _counts(xs, global_steps)
    feature_bounds = _compute_feature_bounds(store, xs)
    outfile = a.outfile or os.path.join(a.log_dir, "state_exploration.png")
    _plot(xs, per_alg_counts, overall_counts, feature_bounds, outfile)


if __name__ == "__main__":
    main()
