from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ALLOWED = {4, 5, 6, 7}
MARKER = "/home/esjung/vllm-ep/utilize.py --worker --gpu "


def _workers() -> list[tuple[int, int, str]]:
    result = subprocess.run(["ps", "-eo", "pid=,uid=,args="], text=True, capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[1]) != os.getuid() or MARKER not in fields[2]:
            continue
        try:
            gpu = int(fields[2].split(MARKER, 1)[1].split()[0])
        except (IndexError, ValueError):
            continue
        if gpu in ALLOWED:
            rows.append((int(fields[0]), gpu, fields[2]))
    return rows


def _descendants(roots: list[int]) -> list[int]:
    result = subprocess.run(["ps", "-eo", "pid=,ppid=,uid="], text=True, capture_output=True, check=True)
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and int(fields[2]) == os.getuid():
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    found, stack = [], list(roots)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.append(child)
                stack.append(child)
    return found


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def stop(gpus: set[int], record: Path) -> None:
    rows = [row for row in _workers() if row[1] in gpus]
    roots = [row[0] for row in rows]
    children = _descendants(roots)
    for pid in list(reversed(children)) + roots:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in roots + children):
        time.sleep(1)
    for pid in roots + children:
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    _write(record, {"action": "stop", "gpus": sorted(gpus), "workers": rows, "remaining": _workers()})


def start(gpus: set[int], record: Path) -> None:
    active = {gpu for _, gpu, _ in _workers()}
    missing = sorted(gpus - active)
    if missing:
        env = dict(os.environ)
        env["VLLM_UTILIZE_GPUS"] = ",".join(map(str, missing))
        log = record.with_suffix(".log")
        with log.open("ab") as handle:
            subprocess.Popen(
                ["bash", "/home/esjung/vllm-ep/run_utilize.sh"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        time.sleep(8)
    _write(record, {"action": "start", "gpus": sorted(gpus), "requested_missing": missing, "workers": _workers()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stop", "start", "status"))
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    gpus = {int(value) for value in args.gpus.split(",") if value}
    if not gpus or not gpus <= ALLOWED:
        raise ValueError(gpus)
    if args.action == "stop":
        stop(gpus, args.record)
    elif args.action == "start":
        start(gpus, args.record)
    else:
        _write(args.record, {"action": "status", "gpus": sorted(gpus), "workers": _workers()})


if __name__ == "__main__":
    main()
