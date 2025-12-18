#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import cv2
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.neighbors import NearestNeighbors


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


class TrajectoryStore:

    def __init__(self, rows, alpha):
        self.rows = rows
        self.total = len(rows)
        self.alpha = float(alpha)
        self.raw = {}
        self.smoothed = {}
        self.starts = {}
        self._build()

    def _build(self):
        raw = {}
        smooth = {}
        starts = {}
        prev = {}
        prev_end = {}
        for row in self.rows:
            alg = int(row["alg_idx"])
            obs = _to_np(row["observation"])
            end_flag = _flag_episode_end(row.get("episode_end"))
            start = prev.get(alg) is None or prev_end.get(alg, True)
            raw.setdefault(alg, []).append(obs)
            if start or self.alpha == 0.0:
                sm_val = obs
            else:
                sm_val = self.alpha * prev[alg] + obs
            smooth.setdefault(alg, []).append(sm_val)
            starts.setdefault(alg, []).append(start)
            prev[alg] = sm_val
            prev_end[alg] = end_flag
        self.raw = {k: np.stack(v) for k, v in raw.items()}
        self.smoothed = {k: np.stack(v) for k, v in smooth.items()}
        self.starts = {k: np.array(v, dtype=bool) for k, v in starts.items()}

    def prefix(self, count):
        count = max(0, min(count, self.total))
        if self.total == 0:
            return {}, {}, {}
        progress = count / self.total
        raw = {}
        smooth = {}
        starts = {}
        for alg in self.raw.keys():
            length = len(self.raw[alg])
            if length == 0:
                continue
            upto = int(np.ceil(length * progress - 1e-12))
            if upto <= 0:
                continue
            upto = min(length, upto)
            raw[alg] = self.raw[alg][:upto]
            smooth[alg] = self.smoothed[alg][:upto]
            starts[alg] = self.starts[alg][:upto]
        return raw, smooth, starts


class Projection:
    uses_smoothed = True

    def __init__(self, name):
        self.name = name
        self.full_view = None
        self.limits = None

    def fit(self, store: TrajectoryStore):
        raise NotImplementedError

    def _finalize(self, trajs):
        self.full_view = trajs or {}
        self.limits = _compute_limits(
            self.full_view) if self.full_view else None

    def view(self, raw_subset, smooth_subset):
        source = smooth_subset if self.uses_smoothed else raw_subset
        if not source or not self.full_view:
            return None
        out = {}
        for alg, arr in source.items():
            full = self.full_view.get(alg)
            if full is None:
                continue
            out[alg] = full[:len(arr)]
        return out or None


class PcaProjection(Projection):

    def __init__(self):
        super().__init__("PCA")
        self.model = None

    def fit(self, store):
        trajs = store.smoothed
        if not trajs:
            self._finalize({})
            return
        print("Fitting PCA on full trajectories...")
        stacked = np.concatenate(list(trajs.values()), 0)
        self.model = PCA(n_components=2).fit(stacked)
        print("PCA ready.")
        full = {k: self.model.transform(v) for k, v in trajs.items()}
        self._finalize(full)


class TsneProjection(Projection):

    def __init__(self, perp):
        super().__init__(f"t-SNE p={perp}")
        self.perp = float(perp)

    def fit(self, store):
        trajs = store.smoothed
        data = []
        order = []
        for alg in sorted(trajs):
            arr = trajs[alg]
            if len(arr) == 0:
                continue
            data.append(arr)
            order.append((alg, len(arr)))
        if not data:
            self._finalize({})
            print(f"t-SNE p={self.perp} skipped (no data).")
            return
        stacked = np.concatenate(data, 0)
        if stacked.shape[0] < 2:
            self._finalize({})
            print(f"t-SNE p={self.perp} skipped (insufficient points).")
            return
        max_perp = max(1.0, stacked.shape[0] - 1)
        perp = max(1.0, min(self.perp, max_perp))
        print(f"Fitting t-SNE (perplexity={self.perp})...")
        emb = TSNE(n_components=2,
                   perplexity=perp,
                   init="random",
                   learning_rate="auto").fit_transform(stacked)
        result = {}
        offset = 0
        for alg, length in order:
            result[alg] = emb[offset:offset + length]
            offset += length
        self._finalize(result)
        if result:
            print(f"t-SNE p={self.perp} ready.")


class LaplacianProjection(Projection):
    uses_smoothed = False

    def __init__(self, eps, temporal_weight):
        super().__init__("Laplacian")
        self.eps = eps
        self.temporal_weight = temporal_weight

    def fit(self, store):
        raw = store.raw
        starts = store.starts
        print("Fitting Laplacian embedding...")
        result = _build_laplacian(raw, starts, self.eps, self.temporal_weight)
        if result:
            print("Laplacian embedding ready.")
        else:
            print("Laplacian embedding skipped (needs >=3 points).")
        self._finalize(result)


def _normalize_states(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1e-6
    return (arr - mean) / std


def _strong_edges(arr, eps):
    nbrs = NearestNeighbors(radius=eps, metric="chebyshev")
    nbrs.fit(arr)
    graph = nbrs.radius_neighbors_graph(arr, mode="connectivity")
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph.maximum(graph.T) if graph.nnz else graph


def _temporal_edges(order, n, weight):
    if weight <= 0:
        return csr_matrix((n, n))
    rows = []
    cols = []
    vals = []
    offset = 0
    for _, length, flags in order:
        for i in range(1, length):
            if flags[i]:
                continue
            a = offset + i - 1
            b = offset + i
            rows.extend([a, b])
            cols.extend([b, a])
            vals.extend([weight, weight])
        offset += length
    if not rows:
        return csr_matrix((n, n))
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def _build_laplacian(raw_trajs, starts, eps, temporal_weight):
    data = []
    order = []
    for alg in sorted(raw_trajs):
        arr = raw_trajs[alg]
        if len(arr) == 0:
            continue
        flags = starts.get(alg)
        if flags is None or len(flags) != len(arr):
            flags = np.zeros(len(arr), dtype=bool)
            flags[0] = True
        data.append(arr)
        order.append((alg, len(arr), flags))
    if not data:
        return {}
    stacked = np.concatenate(data, 0)
    if stacked.shape[0] < 3:
        return {}
    normalized = _normalize_states(stacked)
    strong = _strong_edges(normalized, eps)
    temporal = _temporal_edges(order, stacked.shape[0], temporal_weight)
    graph = strong + temporal
    graph = graph + diags(np.ones(stacked.shape[0]))
    emb = SpectralEmbedding(n_components=2,
                            affinity="precomputed").fit_transform(graph)
    result = {}
    offset = 0
    for alg, length, _ in order:
        result[alg] = emb[offset:offset + length]
        offset += length
    return result


def _compute_limits(trajs):
    if not trajs:
        return None
    xs = np.concatenate([t[:, 0] for t in trajs.values()])
    ys = np.concatenate([t[:, 1] for t in trajs.values()])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    dx = max(1e-6, 0.05 * (x1 - x0))
    dy = max(1e-6, 0.05 * (y1 - y0))
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def _plot_one(ax, trajs, decay, title, xlim=None, ylim=None):
    colors = plt.cm.tab10.colors
    rng = np.random.default_rng()
    for alg, t in sorted(trajs.items()):
        if len(t) == 0:
            continue
        x_span = (xlim[1] - xlim[0]) if xlim else (t[:, 0].max() -
                                                   t[:, 0].min())
        y_span = (ylim[1] - ylim[0]) if ylim else (t[:, 1].max() -
                                                   t[:, 1].min())
        x_jit = 0.005 * (x_span if x_span > 0 else 1.0)
        y_jit = 0.005 * (y_span if y_span > 0 else 1.0)
        jitter = np.empty_like(t)
        jitter[:, 0] = t[:, 0] + (rng.random(len(t)) - 0.5) * 2 * x_jit
        jitter[:, 1] = t[:, 1] + (rng.random(len(t)) - 0.5) * 2 * y_jit
        c = colors[alg % len(colors)]
        ax.plot(jitter[:, 0], jitter[:, 1], color=c, lw=1, alpha=0.2)
        n = len(jitter)
        ages = np.arange(n - 1, -1, -1)
        alphas = decay**ages
        ax.scatter(jitter[:, 0],
                   jitter[:, 1],
                   s=10,
                   color=[(c[0], c[1], c[2], a) for a in alphas])
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def _plot_grid(slots, decay, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    flat = axes.flatten()
    for ax, slot in zip(flat, slots):
        if slot is None:
            ax.axis("off")
            continue
        title, trajs, limits = slot
        xlim, ylim = limits if limits else (None, None)
        _plot_one(ax, trajs, decay, title, xlim, ylim)
    for ax in flat[len(slots):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"saved {outfile}")
    plt.close(fig)


def _write_video(image_paths, out_path, fps=1):
    frames = [cv2.imread(p) for p in image_paths if os.path.isfile(p)]
    frames = [f for f in frames if f is not None]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        vw.write(f)
    vw.release()
    print(f"saved {out_path}")


def _make_slots(pca, lap, tsnes, raw, smooth):
    slots = []
    slots.append(_make_slot(pca, raw, smooth))
    slots.append(_make_slot(tsnes[0], raw, smooth) if tsnes else None)
    slots.append(_make_slot(lap, raw, smooth))
    slots.append(_make_slot(tsnes[1], raw, smooth) if len(tsnes) > 1 else None)
    return slots


def _make_slot(proj, raw, smooth):
    if proj is None:
        return None
    view = proj.view(raw, smooth)
    if view is None:
        return None
    return (proj.name, view, proj.limits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--alpha-decay", type=float, default=0.8)
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--snapshots", type=int, default=10)
    p.add_argument("--tsne-perplexities", type=str, default="16,64")
    p.add_argument("--lap-eps", type=float, default=0.01)
    p.add_argument("--lap-temporal-weight", type=float, default=0.5)
    a = p.parse_args()
    tsne_perps = [
        float(x) for x in a.tsne_perplexities.split(",") if x.strip()
    ]
    if len(tsne_perps) > 2:
        print(f"Using first two t-SNE perplexities: {tsne_perps[:2]}")
        tsne_perps = tsne_perps[:2]
    ndjson = os.path.join(a.log_dir, "rollout_states.ndjson")
    if not os.path.isfile(ndjson):
        print(f"missing {ndjson}", file=sys.stderr)
        sys.exit(1)
    rows = _rows(_load_table(ndjson))
    if not rows:
        print("no trajectories", file=sys.stderr)
        sys.exit(1)
    store = TrajectoryStore(rows, a.alpha)
    if not store.smoothed:
        print("no trajectories", file=sys.stderr)
        sys.exit(1)
    pca = PcaProjection()
    pca.fit(store)
    lap = LaplacianProjection(a.lap_eps, a.lap_temporal_weight)
    lap.fit(store)
    tsne_projs = []
    for perp in tsne_perps:
        proj = TsneProjection(perp)
        proj.fit(store)
        tsne_projs.append(proj)
    outfile = a.outfile or os.path.join(a.log_dir, "trajectories.png")
    snapshot_dir = os.path.dirname(outfile) or "."
    os.makedirs(snapshot_dir, exist_ok=True)
    image_paths = []
    for i in range(1, a.snapshots + 1):
        cutoff = max(1, int(store.total * i / a.snapshots))
        raw_subset, smooth_subset, _ = store.prefix(cutoff)
        slots = _make_slots(pca, lap, tsne_projs, raw_subset, smooth_subset)
        if not any(slots):
            continue
        path = os.path.join(
            snapshot_dir,
            f"{os.path.splitext(os.path.basename(outfile))[0]}_{i:02d}.png")
        _plot_grid(slots, decay=a.alpha_decay, outfile=path)
        image_paths.append(path)
    if image_paths:
        video_path = os.path.join(snapshot_dir, "trajectories_progress.mp4")
        _write_video(image_paths, video_path, fps=1)
        shutil.copyfile(image_paths[-1], outfile)
        print(f"saved {outfile}")
    else:
        print("no snapshots generated", file=sys.stderr)


if __name__ == "__main__":
    main()
