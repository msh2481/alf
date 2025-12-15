# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def _obs_idx(location: int, timestep: int, k: int) -> int:
    return (location + k) * (k + 1) + timestep


def _maybe_list_get0(df: pl.DataFrame, col: str) -> pl.Expr:
    t = df.schema.get(col)
    if isinstance(t, pl.List):
        return pl.col(col).list.get(0)
    return pl.col(col)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--location", type=int, required=True)
    p.add_argument("--timestep", type=int, required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    idx = _obs_idx(args.location, args.timestep, args.k)
    df = pl.read_ndjson(args.path)
    if "obs_idx" not in df.columns:
        raise ValueError("trace file missing 'obs_idx'")

    df = df.filter(pl.col("obs_idx") == idx)
    if df.is_empty():
        raise ValueError(
            f"no rows for obs_idx={idx} (location={args.location}, timestep={args.timestep}, k={args.k})"
        )

    mean0 = _maybe_list_get0(df, "mean").cast(pl.Float64).alias("mean0")
    std0 = _maybe_list_get0(df, "std").cast(pl.Float64).alias("std0")
    dqda0 = _maybe_list_get0(df, "dqda").cast(pl.Float64).alias("dqda0")
    q_plus = _maybe_list_get0(df, "q_plus").cast(pl.Float64).alias("q_plus")
    q_minus = _maybe_list_get0(df, "q_minus").cast(pl.Float64).alias("q_minus")

    df = df.with_columns([
        pl.col("update_id").cast(pl.Int64),
        mean0,
        std0,
        dqda0,
        q_plus,
        q_minus,
    ])

    ts = df.group_by("update_id").agg([
        pl.len().alias("n"),
        pl.col("q_plus").mean().alias("q_plus_mean"),
        pl.col("q_minus").mean().alias("q_minus_mean"),
        pl.col("mean0").mean().alias("actor_mean"),
        pl.col("dqda0").mean().alias("dqda_mean"),
    ]).sort("update_id")

    x = ts["update_id"].to_numpy()
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    fig.suptitle(
        f"Trace cell: location={args.location}, timestep={args.timestep}, k={args.k}, obs_idx={idx}\n{args.path}"
    )

    axs[0].plot(x, ts["q_plus_mean"].to_numpy(), label="Q(s,+1)")
    axs[0].plot(x, ts["q_minus_mean"].to_numpy(), label="Q(s,-1)")
    axs[0].set_ylabel("Q")
    axs[0].legend()

    axs[1].plot(x, ts["actor_mean"].to_numpy(), label="actor mean")
    axs[1].set_ylabel("actor")
    axs[1].legend()

    axs[2].plot(x, ts["dqda_mean"].to_numpy(), label="dQ/da mean")
    axs[2].set_ylabel("dQ/da")
    axs[2].legend()

    dq = ts["q_plus_mean"] - ts["q_minus_mean"]
    axs[3].plot(x, dq.to_numpy(), label="Q(s,+1)-Q(s,-1)")
    axs[3].axhline(0.0, linewidth=1)
    axs[3].set_ylabel("ΔQ")
    axs[3].set_xlabel("update_id")
    axs[3].legend()

    fig.tight_layout()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
