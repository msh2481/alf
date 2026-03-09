#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
USE_COLOR = sys.stdout.isatty()
RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


@dataclass(frozen=True)
class ExportTask:
    name: str
    out_dir: str
    command: list[str]
    target_return: float = 1000.0


def _style(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{RESET}"


def _warning(text: str) -> str:
    return _style(text, YELLOW)


def _success(text: str) -> str:
    return _style(text, GREEN)


TASKS: dict[str, ExportTask] = {
    "one_vs_many_synthetic":
    ExportTask(
        name="one_vs_many_synthetic",
        out_dir="results/one_vs_many_synthetic",
        command=[
            "python",
            "tools/custom_plot.py",
            "--folder",
            "/tmp/bipolar/BipolarChain-medium-sparse-onehot-discrete-v0/",
            "--episode_index_base_agents",
            "4",
            "--names",
            "a1_e4_prior0.0_noreset",
            "a1_e4_prior2.0_noreset",
            "a4_e4_prior0.0",
            "a4_e4_prior2.0",
        ],
        target_return=1.0,
    ),
    "prior_vs_no_prior_synthetic":
    ExportTask(
        name="prior_vs_no_prior_synthetic",
        out_dir="results/prior_vs_no_prior_synthetic",
        command=[
            "python",
            "tools/custom_plot.py",
            "--folder",
            "/tmp/bipolar/BipolarChain-medium-sparse-onehot-discrete-v0/",
            "--episode_index_base_agents",
            "4",
            "--names",
            "with_prior",
            "without_prior",
        ],
        target_return=1.0,
    ),
    "prior_vs_no_prior_actor_dmc_four_agents_soft_reset":
    ExportTask(
        name="prior_vs_no_prior_actor_dmc_four_agents_soft_reset",
        out_dir="results/prior_vs_no_prior_actor_dmc_four_agents_soft_reset",
        command=[
            "python",
            "tools/custom_plot.py",
            "--folder",
            "all_dm",
            "--episode_index_base_agents",
            "4",
            "--names",
            "a4_actorprior0",
            "a4_actorprior0.001",
            "a4_actorprior0.01",
            "a4_actorprior0.1",
            "a4_actorprior1.0",
        ],
    ),
    "prior_vs_no_prior_dmc":
    ExportTask(
        name="prior_vs_no_prior_dmc",
        out_dir="results/prior_vs_no_prior_dmc",
        command=[
            "python",
            "tools/custom_plot.py",
            "--folder",
            "all_dm",
            "--episode_index_base_agents",
            "4",
            "--names",
            "a1_prior0",
            "a1_prior0.1",
            "a1_prior0.01",
            "a1_prior0.001",
        ],
    ),
    "prior_vs_no_prior_dmc_single_layer":
    ExportTask(
        name="prior_vs_no_prior_dmc_single_layer",
        out_dir="results/prior_vs_no_prior_dmc_single_layer",
        command=[
            "python",
            "tools/custom_plot.py",
            "--folder",
            "all_dm",
            "--episode_index_base_agents",
            "4",
            "--names",
            "a1_single_layer_prior0",
            "a1_single_layer_prior0.1",
            "a1_single_layer_prior0.01",
            "a1_single_layer_prior0.001",
        ],
    ),
    "prior_vs_no_prior_dmc_four_agents_soft_reset":
    ExportTask(
        name="prior_vs_no_prior_dmc_four_agents_soft_reset",
        out_dir="results/prior_vs_no_prior_dmc_four_agents_soft_reset",
        command=[
            "python",
            "tools/custom_plot.py",
            "--folder",
            "all_dm",
            "--episode_index_base_agents",
            "4",
            "--names",
            "a4_prior0",
            "a4_prior0.001",
            "a4_prior0.01",
            "a4_prior0.1",
            "a4_prior1.0",
        ],
    ),
}


def _with_out_dir(command: Sequence[str], out_dir: str) -> list[str]:
    return [*command, "--out_dir", out_dir]


def _flag_value(command: Sequence[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def _flag_values(command: Sequence[str], flag: str) -> list[str]:
    idx = command.index(flag) + 1
    values: list[str] = []
    while idx < len(command) and not command[idx].startswith("--"):
        values.append(command[idx])
        idx += 1
    return values


def _out_dir_path(task: ExportTask) -> Path:
    return (REPO_ROOT / task.out_dir).resolve()


def _plot_command(task: ExportTask) -> list[str]:
    return _with_out_dir(task.command, str(_out_dir_path(task)))


def _regret_command(task: ExportTask) -> list[str]:
    command = list(task.command)
    command[1] = "tools/custom_regret_table.py"
    return [
        *_with_out_dir(command, str(_out_dir_path(task))),
        "--target_return",
        str(task.target_return),
    ]


def _plot_command_string(task: ExportTask) -> str:
    return shlex.join(_plot_command(task))


def _regret_command_string(task: ExportTask) -> str:
    return shlex.join(_regret_command(task))


def _combined_command_string(task: ExportTask) -> str:
    out_dir = _out_dir_path(task)
    if out_dir.exists():
        return _regret_command_string(task)
    return f"{_plot_command_string(task)} && {_regret_command_string(task)}"


def _run_local(task: ExportTask) -> None:
    out_dir = _out_dir_path(task)
    if out_dir.exists():
        print(
            _warning(
                f"warning: skipping plots for {task.name} because {out_dir} exists"
            ))
    else:
        subprocess.run(_plot_command(task), cwd=REPO_ROOT, check=True)
    subprocess.run(_regret_command(task), cwd=REPO_ROOT, check=True)


def _queue_pueue(task: ExportTask) -> str:
    command = _combined_command_string(task)
    return subprocess.run(
        ["pueue", "add", "-p", "--", command],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_plot_folders(task: ExportTask) -> list[Path]:
    folder = _flag_value(task.command, "--folder")
    if folder == "all_dm":
        dmc_root = Path("/tmp/dmc")
        if not dmc_root.exists():
            return []
        return sorted((p for p in dmc_root.iterdir() if p.is_dir()),
                      key=lambda p: p.name)
    return [Path(folder)]


def _format_timestamp(ts: float | None) -> str:
    if ts is None:
        return "missing"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _format_timestamp_range(values: Sequence[float | None]) -> str:
    present = sorted(v for v in values if v is not None)
    if not present:
        return "missing"
    lo = _format_timestamp(present[0])
    hi = _format_timestamp(present[-1])
    return lo if lo == hi else f"{lo} .. {hi}"


def _latest_events_mtime(run_root: Path) -> tuple[float | None, float | None, int]:
    if not run_root.exists():
        return None, None, 0

    event_files = sorted(run_root.glob("*/events.ndjson"))
    if not event_files:
        event_files = sorted(run_root.rglob("events.ndjson"))
    if not event_files:
        return None, None, 0

    latest = max(path.stat().st_mtime for path in event_files)
    seed0 = run_root / "0" / "events.ndjson"
    seed0_mtime = seed0.stat().st_mtime if seed0.exists() else None
    return latest, seed0_mtime, len(event_files)


def _print_source_freshness(task: ExportTask) -> None:
    names = _flag_values(task.command, "--names")
    folders = _resolve_plot_folders(task)
    folder_arg = _flag_value(task.command, "--folder")

    print("source freshness:")
    if not folders:
        if folder_arg == "all_dm":
            print("  /tmp/dmc: missing")
        else:
            print(f"  {folder_arg}: missing")
        return

    if folder_arg == "all_dm":
        total_envs = len(folders)
        for run_name in names:
            latest_values: list[float | None] = []
            seed0_values: list[float | None] = []
            present_envs = 0
            total_events = 0
            for folder in folders:
                run_root = folder / run_name
                latest, seed0, events_count = _latest_events_mtime(run_root)
                latest_values.append(latest)
                seed0_values.append(seed0)
                if events_count > 0:
                    present_envs += 1
                    total_events += events_count
            print(
                f"  {run_name}: latest={_format_timestamp_range(latest_values)} "
                f"seed0={_format_timestamp_range(seed0_values)} "
                f"envs={present_envs}/{total_envs} events={total_events}")
        return

    for folder in folders:
        for run_name in names:
            run_root = folder / run_name
            latest, seed0, events_count = _latest_events_mtime(run_root)
            print(
                f"  {folder.name}/{run_name}: latest={_format_timestamp(latest)} "
                f"seed0={_format_timestamp(seed0)} events={events_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export hardcoded experiment plots into results/.")
    parser.add_argument(
        "tasks",
        nargs="*",
        choices=sorted(TASKS.keys()),
        help="Tasks to export. If omitted, all hardcoded tasks are used.",
    )
    parser.add_argument(
        "--runner",
        choices=("pueue", "local"),
        default="pueue",
        help="How to execute plotting commands.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.tasks or list(TASKS.keys())

    for key in selected:
        task = TASKS[key]
        out_dir = _out_dir_path(task)
        print(f"[{task.name}] {task.out_dir}")
        _print_source_freshness(task)
        if out_dir.exists():
            print(
                _warning(
                    f"warning: skipping plots for {task.name} because {out_dir} exists"
                ))
        print(_combined_command_string(task))
        if args.print_only:
            print()
            continue
        if args.runner == "pueue":
            task_id = _queue_pueue(task)
            print(_success(f"queued as {task_id}"))
        else:
            _run_local(task)
            print(_success("completed"))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
