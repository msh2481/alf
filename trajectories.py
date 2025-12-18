#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, shutil, math
import numpy as np
import matplotlib.pyplot as plt


def _load_table(path):
    try:
        import polars as pl
        return pl.read_ndjson(path)
    except ImportError:
        import pandas as pd
        return pd.read_json(path, lines=True)


def _rows(table):
    if "to_dicts" in dir(table):
        return table.sort(["rollout_step", "alg_idx"]).to_dicts()
    return table.sort_values(["rollout_step", "alg_idx"]).to_dict("records")


def _to_np(obs):
    arr = np.asarray(obs, dtype=np.float64)
    return arr.reshape(-1) if arr.ndim > 1 else arr


def _smooth(traj, alpha):
    if len(traj) == 0:
        return traj
    out = np.zeros_like(traj)
    out[0] = traj[0]
    for i in range(1, len(traj)):
        out[i] = alpha * out[i - 1] + traj[i]
    return out


def _collect(rows, alpha):
    data = {}
    for r in rows:
        alg = int(r["alg_idx"])
        data.setdefault(alg, []).append(_to_np(r["observation"]))
    return {k: _smooth(np.stack(v, 0), alpha) for k, v in data.items() if v}


def _fit_pca(trajs):
    from sklearn.decomposition import PCA
    stacked = np.concatenate(list(trajs.values()), 0)
    return PCA(n_components=2).fit(stacked)


def _apply_pca(pca, trajs):
    return {k: pca.transform(v) for k, v in trajs.items()}


def _fit_tsne(trajs, perplexity):
    from sklearn.manifold import TSNE
    data = []
    order = []
    for alg in sorted(trajs.keys()):
        arr = trajs[alg]
        if len(arr) == 0:
            continue
        data.append(arr)
        order.append((alg, len(arr)))
    if not data:
        return None
    stacked = np.concatenate(data, 0)
    if stacked.shape[0] < 2:
        return None
    max_perp = max(1.0, stacked.shape[0] - 1)
    perp = max(1.0, min(perplexity, max_perp))
    tsne = TSNE(n_components=2,
                perplexity=perp,
                init="random",
                learning_rate="auto")
    emb = tsne.fit_transform(stacked)
    result = {}
    offset = 0
    for alg, length in order:
        result[alg] = emb[offset:offset + length]
        offset += length
    return result


def _slice_view(full_view, subset_trajs):
    sliced = {}
    for alg, traj in subset_trajs.items():
        full = full_view.get(alg)
        if full is None:
            continue
        length = len(traj)
        sliced[alg] = full[:length]
    return sliced


def _compute_limits(trajs):
    xs = np.concatenate([t[:, 0] for t in trajs.values()])
    ys = np.concatenate([t[:, 1] for t in trajs.values()])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    dx = max(1e-6, 0.05 * (x1 - x0))
    dy = max(1e-6, 0.05 * (y1 - y0))
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def _plot_one(ax, trajs, decay, title, xlim=None, ylim=None):
    colors = plt.cm.tab10.colors
    for alg, t in sorted(trajs.items()):
        if len(t) == 0:
            continue
        c = colors[alg % len(colors)]
        ax.plot(t[:, 0], t[:, 1], color=c, lw=1, alpha=0.2, label=f"alg {alg}")
        n = len(t)
        ages = np.arange(n - 1, -1, -1)
        alphas = decay**ages
        ax.scatter(t[:, 0],
                   t[:, 1],
                   s=10,
                   color=[(c[0], c[1], c[2], a) for a in alphas])
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def _plot(views, decay, outfile):
    if not views:
        return
    cols = 2 if len(views) > 1 else 1
    rows = math.ceil(len(views) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (title, trajs, limits) in zip(axes, views):
        xlim, ylim = (limits if limits else (None, None))
        _plot_one(ax, trajs, decay, title, xlim=xlim, ylim=ylim)
    for ax in axes[len(views):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"saved {outfile}")
    plt.close(fig)


def _write_video(image_paths, out_path, fps=1):
    import cv2
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
    p.add_argument("--tsne-perplexities",
                   type=str,
                   default="16,32,64",
                   help="Comma-separated perplexities for t-SNE views")
    a = p.parse_args()
    tsne_perps = [
        float(x) for x in a.tsne_perplexities.split(",") if x.strip()
    ] or [30.0]
    ndjson = os.path.join(a.log_dir, "rollout_states.ndjson")
    if not os.path.isfile(ndjson):
        print(f"missing {ndjson}", file=sys.stderr)
        sys.exit(1)
    rows = _rows(_load_table(ndjson))
    if not rows:
        print("no trajectories", file=sys.stderr)
        sys.exit(1)
    final_trajs = _collect(rows, alpha=a.alpha)
    if not final_trajs:
        print("no trajectories", file=sys.stderr)
        sys.exit(1)
    print("Fitting PCA on full trajectories...")
    pca = _fit_pca(final_trajs)
    print("PCA ready.")
    outfile = a.outfile or os.path.join(a.log_dir, "trajectories.png")
    snapshot_dir = os.path.dirname(outfile) or "."
    os.makedirs(snapshot_dir, exist_ok=True)
    total = len(rows)
    final_pca = _apply_pca(pca, final_trajs)
    pca_limits = _compute_limits(final_pca)
    tsne_final = {}
    tsne_limits = {}
    for perp in tsne_perps:
        print(f"Fitting t-SNE (perplexity={perp})...")
        view = _fit_tsne(final_trajs, perp)
        if view:
            tsne_final[perp] = view
            tsne_limits[perp] = _compute_limits(view)
            print(f"t-SNE p={perp} ready.")
        else:
            print(f"t-SNE p={perp} skipped (insufficient data).")
    image_paths = []
    for i in range(1, a.snapshots + 1):
        cutoff = max(1, int(total * i / a.snapshots))
        subset = rows[:cutoff]
        trajs = _collect(subset, alpha=a.alpha)
        if not trajs:
            continue
        pca_trajs = _apply_pca(pca, trajs)
        views = [("PCA", pca_trajs, pca_limits)]
        for perp in tsne_perps:
            full_view = tsne_final.get(perp)
            if not full_view:
                continue
            tsne_trajs = _slice_view(full_view, trajs)
            if not tsne_trajs:
                continue
            limits = tsne_limits.get(perp)
            views.append((f"t-SNE p={perp}", tsne_trajs, limits))
        if not views:
            continue
        path = os.path.join(
            snapshot_dir,
            f"{os.path.splitext(os.path.basename(outfile))[0]}_{i:02d}.png")
        _plot(views, decay=a.alpha_decay, outfile=path)
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
