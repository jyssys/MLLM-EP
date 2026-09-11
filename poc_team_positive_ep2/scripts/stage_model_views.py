#!/usr/bin/env python3
"""Create immutable baseline/TEAM views over one pinned SDAR snapshot."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_view(snapshot: Path, destination: Path, modeling_file: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        if source.name in {"modeling_sdar_moe.py", ".cache"}:
            continue
        target = destination / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    shutil.copy2(modeling_file, destination / "modeling_sdar_moe.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--team-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    baseline_source = snapshot / "modeling_sdar_moe.py"
    team_source = args.team_repo.resolve() / "modeling_sdar_moe.py"
    if not baseline_source.is_file() or not team_source.is_file():
        raise FileNotFoundError("baseline or TEAM modeling file is missing")

    baseline_view = args.output / "SDAR_BASELINE"
    team_view = args.output / "TEAM_METHOD"
    make_view(snapshot, baseline_view, baseline_source)
    make_view(snapshot, team_view, team_source)

    print(f"baseline={baseline_view} sha256={sha256(baseline_view / 'modeling_sdar_moe.py')}")
    print(f"team={team_view} sha256={sha256(team_view / 'modeling_sdar_moe.py')}")
    print(f"snapshot={snapshot}")


if __name__ == "__main__":
    main()
