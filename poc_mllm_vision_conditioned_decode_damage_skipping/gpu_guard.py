"""Stop/restart only this user's configured GPU4--7 utilization workers."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ALLOWED = {4, 5, 6, 7}


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def workers() -> list[tuple[int, int, str]]:
    result = subprocess.run(["ps", "-eo", "pid=,uid=,args="], text=True, capture_output=True, check=True)
    marker = "/home/esjung/vllm-ep/utilize.py --worker --gpu "
    answer = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[1]) != os.getuid() or marker not in fields[2]:
            continue
        try:
            gpu = int(fields[2].split(marker, 1)[1].split()[0])
        except (IndexError, ValueError):
            continue
        if gpu in ALLOWED:
            answer.append((int(fields[0]), gpu, fields[2]))
    return answer


def controllers() -> list[tuple[int, list[int], str]]:
    result = subprocess.run(["ps", "-eo", "pid=,uid=,args="], text=True, capture_output=True, check=True)
    marker = "/home/esjung/vllm-ep/utilize.py --gpus "
    answer = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[1]) != os.getuid() or marker not in fields[2]:
            continue
        try:
            values = fields[2].split(marker, 1)[1].split()[0]
            gpus = [int(value) for value in values.split(",")]
        except (IndexError, ValueError):
            continue
        if set(gpus) & ALLOWED:
            answer.append((int(fields[0]), gpus, fields[2]))
    return answer


def descendants(roots: list[int]) -> list[int]:
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


def stop(record: Path) -> None:
    rows = workers()
    parent_rows = controllers()
    roots = [pid for pid, _, _ in rows] + [pid for pid, _, _ in parent_rows]
    children = descendants(roots)
    for pid in list(reversed(children)) + roots:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in roots + children):
        time.sleep(1)
    for pid in [pid for pid in roots + children if Path(f"/proc/{pid}").exists()]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    write(record, {
        "action": "stop", "workers": rows, "controllers": parent_rows,
        "descendants": children, "remaining_workers": workers(),
        "remaining_controllers": controllers(),
    })


def start(record: Path) -> None:
    if workers() or controllers():
        write(record, {"action": "already_running", "workers": workers(), "controllers": controllers()})
        return
    env = dict(os.environ)
    env["VLLM_UTILIZE_GPUS"] = "4,5,6,7"
    log = record.with_suffix(".log")
    with log.open("ab") as handle:
        subprocess.Popen(
            ["bash", "/home/esjung/vllm-ep/run_utilize.sh"], stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True, env=env,
        )
    time.sleep(5)
    write(record, {"action": "start", "workers": workers(), "log": str(log)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stop", "start", "status"))
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "stop":
        stop(args.record)
    elif args.action == "start":
        start(args.record)
    else:
        write(args.record, {"action": "status", "workers": workers()})


if __name__ == "__main__":
    main()
