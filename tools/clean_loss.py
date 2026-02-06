#!/usr/bin/env python3
"""Remove loss lines from DMC ndjson logs.

Walks `/tmp/dmc/*/*/*/events.ndjson` and removes lines whose (left-stripped)
text starts with the exact prefix:

    {"type": "loss"

Edits files in-place using an atomic replace (write temp -> os.replace).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


LOSS_PREFIX = '{"type": "loss"'


@dataclass(frozen=True)
class FileResult:
    path: Path
    removed: int
    kept: int


def _iter_event_files(root: Path) -> list[Path]:
    # matches: /tmp/dmc/<env>/<exp>/<seed>/events.ndjson
    return sorted(root.glob("*/*/*/events.ndjson"))


def _clean_one_file(path: Path, *, dry_run: bool, backup: bool) -> FileResult:
    print(f"Cleaning {path}...")
    removed = 0
    kept = 0

    # First pass: count + optionally write filtered content.
    if dry_run:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith(LOSS_PREFIX):
                    removed += 1
                else:
                    kept += 1
        return FileResult(path=path, removed=removed, kept=kept)

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="events.", suffix=".ndjson",
                                            dir=str(path.parent))
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as out, path.open(
                "r", encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith(LOSS_PREFIX):
                    removed += 1
                    continue
                kept += 1
                out.write(line)
        tmp_fd = None
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return FileResult(path=path, removed=removed, kept=kept)


def main() -> None:
    p = argparse.ArgumentParser(
        description=
        "Remove ndjson lines starting with '{\"type\": \"loss\"' from /tmp/dmc logs."
    )
    p.add_argument("--root",
                   type=str,
                   default="/tmp/dmc",
                   help="Root directory containing env folders (default: /tmp/dmc).")
    p.add_argument("--dry_run",
                   action="store_true",
                   help="Only report counts; do not modify files.")
    p.add_argument("--backup",
                   action="store_true",
                   help="Write one-time .bak next to each edited file.")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    files = _iter_event_files(root)
    if not files:
        print(f"No files found under: {root}")
        return

    total_removed = 0
    total_files_changed = 0
    total_files = 0

    for path in files:
        total_files += 1
        res = _clean_one_file(path, dry_run=args.dry_run, backup=args.backup)
        total_removed += res.removed
        if res.removed:
            total_files_changed += 1
        print(f"{res.path}: removed={res.removed} kept={res.kept}")

    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(
        f"\n{mode}: files={total_files} changed={total_files_changed} removed_lines={total_removed}"
    )


if __name__ == "__main__":
    main()
