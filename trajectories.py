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

    def __init__(self, rows, alpha, thin=1, max_steps=None):
        self.rows = rows
        self.alpha = float(alpha)
        self.thin = max(1, int(thin))
        self.max_steps = int(max_steps) if max_steps else None
        self.raw = {}
        self.smoothed = {}
        self.starts = {}
        self.ends = {}
        self.total = 0
        self._build()

    def _build(self):
        raw = {}
        smooth = {}
        starts = {}
        ends = {}
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
            ends.setdefault(alg, []).append(end_flag)
            prev[alg] = sm_val
            prev_end[alg] = end_flag
        total = 0
        out_raw = {}
        out_smooth = {}
        out_starts = {}
        out_ends = {}
        for alg in raw.keys():
            r = np.stack(raw[alg])
            s = np.stack(smooth[alg])
            st = np.array(starts[alg], dtype=bool)
            en = np.array(ends[alg], dtype=bool)
            if self.thin > 1 and len(r) > 0:
                mask = np.zeros(len(r), dtype=bool)
                mask[::self.thin] = True
                mask |= st
                mask |= en
                mask[-1] = True
                r = r[mask]
                s = s[mask]
                st = st[mask]
                en = en[mask]
            if self.max_steps is not None and self.max_steps > 0:
                r = r[:self.max_steps]
                s = s[:self.max_steps]
                st = st[:self.max_steps]
                en = en[:self.max_steps]
            out_raw[alg] = r
            out_smooth[alg] = s
            out_starts[alg] = st
            out_ends[alg] = en
            total += len(r)
        self.raw = out_raw
        self.smoothed = out_smooth
        self.starts = out_starts
        self.ends = out_ends
        self.total = total

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

    def __init__(self, knn, temporal_weight):
        super().__init__("Laplacian")
        self.knn = knn
        self.temporal_weight = temporal_weight

    def fit(self, store):
        raw = store.raw
        starts = store.starts
        print("Fitting Laplacian embedding...")
        result = _build_laplacian(raw, starts, self.knn, self.temporal_weight)
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


def _strong_edges(arr, k):
    n = arr.shape[0]
    if n == 0:
        return csr_matrix((0, 0))
    if n == 1:
        return csr_matrix((1, 1))
    k = max(1, min(int(k), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(arr)
    distances, indices = nbrs.kneighbors(arr, return_distance=True)
    rows = []
    cols = []
    for i in range(n):
        neighbors = indices[i, 1:]
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors)
    data = np.ones(len(rows), dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)
    return graph


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


def _start_edges(order, n):
    start_nodes = []
    offset = 0
    for _, length, flags in order:
        for i in range(length):
            if flags[i]:
                start_nodes.append(offset + i)
        offset += length
    if len(start_nodes) < 2:
        return csr_matrix((n, n))
    starts = np.array(start_nodes, dtype=np.int64)
    i_idx, j_idx = np.triu_indices(len(starts), k=1)
    if len(i_idx) == 0:
        return csr_matrix((n, n))
    rows = starts[i_idx]
    cols = starts[j_idx]
    data = np.ones(len(rows), dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    return graph + graph.T


def _build_laplacian(raw_trajs, starts, knn, temporal_weight):
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
    strong = _strong_edges(normalized, knn)
    temporal = _temporal_edges(order, stacked.shape[0], temporal_weight)
    start_graph = _start_edges(order, stacked.shape[0])
    graph = strong + temporal + start_graph
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


def _make_view(proj, raw, smooth):
    if proj is None:
        return None
    view = proj.view(raw, smooth)
    if not view:
        return None
    return (proj.name, view, proj.limits)


def _plot_view(view, decay, outfile):
    if view is None:
        return
    title, trajs, limits = view
    xlim, ylim = limits if limits else (None, None)
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_one(ax, trajs, decay, title, xlim=xlim, ylim=ylim)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--alpha-decay", type=float, default=0.8)
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--snapshots", type=int, default=10)
    p.add_argument("--tsne-perplexities", type=str, default="16,64")
    p.add_argument(
        "--lap-knn",
        type=int,
        default=10,
        help="Number of nearest neighbors for Laplacian graph (strong edges)")
    p.add_argument("--lap-temporal-weight", type=float, default=3.0)
    p.add_argument(
        "--thin",
        type=int,
        default=4,
        help=
        "Keep every Nth trajectory point (>=1). Episode starts/ends always kept."
    )
    p.add_argument("--k",
                   type=int,
                   default=None,
                   help="Limit each agent to first K steps (after thinning).")
    p.add_argument("--view",
                   choices=["pca", "laplacian", "tsne"],
                   default="laplacian",
                   help="Projection to visualize for snapshots/video.")
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
    store = TrajectoryStore(rows, a.alpha, thin=a.thin, max_steps=a.k)
    if not store.smoothed:
        print("no trajectories", file=sys.stderr)
        sys.exit(1)

    active_proj = None
    if a.view == "pca":
        active_proj = PcaProjection()
    elif a.view == "laplacian":
        active_proj = LaplacianProjection(a.lap_knn, a.lap_temporal_weight)
    else:
        perp = tsne_perps[0] if tsne_perps else 30.0
        active_proj = TsneProjection(perp)

    active_proj.fit(store)
    if not active_proj.full_view:
        print(f"{a.view} view unavailable.", file=sys.stderr)
        sys.exit(1)

    outfile = a.outfile or os.path.join(a.log_dir, "trajectories.png")
    snapshot_dir = os.path.dirname(outfile) or "."
    os.makedirs(snapshot_dir, exist_ok=True)

    image_paths = []
    for i in range(1, a.snapshots + 1):
        cutoff = max(1, int(store.total * i / a.snapshots))
        raw_subset, smooth_subset, _ = store.prefix(cutoff)
        slot = _make_view(active_proj, raw_subset, smooth_subset)
        if slot is None:
            continue
        path = os.path.join(
            snapshot_dir,
            f"{os.path.splitext(os.path.basename(outfile))[0]}_{i:02d}.png")
        _plot_view(slot, decay=a.alpha_decay, outfile=path)
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
