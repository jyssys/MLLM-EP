from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .contract import ALLOWED_GPUS

MARKER = "/home/esjung/vllm-ep/utilize.py --worker --gpu "


def _workers() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,uid=,args="], text=True, capture_output=True, check=True
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[1]) != os.getuid() or MARKER not in fields[2]:
            continue
        try:
            gpu = int(fields[2].split(MARKER, 1)[1].split()[0])
        except (IndexError, ValueError):
            continue
        if gpu in ALLOWED_GPUS:
            rows.append((int(fields[0]), gpu, fields[2]))
    return rows


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _owned_gpu_processes(gpus: set[int]) -> list[tuple[int, int, str]]:
    """Resolve current same-user compute processes on the explicitly scoped GPUs.

    vLLM changes burner worker process titles to ``VLLM::EngineCore`` and may
    orphan them when the Python launcher is terminated, so argv matching alone
    cannot guarantee a clean timing/capture boundary.
    """
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    by_uuid = {
        fields[1].strip(): int(fields[0].strip())
        for row in gpu_rows if len(fields := row.split(",", 1)) == 2
    }
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    result = []
    for row in apps:
        fields = row.split(",", 1)
        if len(fields) != 2 or fields[1].strip() not in by_uuid:
            continue
        pid, gpu = int(fields[0].strip()), by_uuid[fields[1].strip()]
        if gpu not in gpus:
            continue
        proc = Path(f"/proc/{pid}")
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        result.append((pid, gpu, cmd))
    return result


def stop(gpus: set[int], record: Path) -> None:
    rows = [row for row in _workers() if row[1] in gpus]
    known = {row[0] for row in rows}
    rows.extend(row for row in _owned_gpu_processes(gpus) if row[0] not in known)
    for pid, _, _ in rows:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid, _, _ in rows):
        time.sleep(0.5)
    for pid, _, _ in rows:
        if Path(f"/proc/{pid}").exists():
            os.kill(pid, signal.SIGKILL)
    _write(record, {"action": "stop", "gpus": sorted(gpus), "workers": rows,
                    "remaining": _workers()})


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
                stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        time.sleep(10)
    _write(record, {"action": "start", "gpus": sorted(gpus),
                    "requested_missing": missing, "workers": _workers()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stop", "start", "status"))
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    gpus = {int(value) for value in args.gpus.split(",") if value}
    if not gpus or not gpus <= ALLOWED_GPUS:
        raise ValueError(f"GPU scope violation: {sorted(gpus)}")
    if args.action == "stop":
        stop(gpus, args.record)
    elif args.action == "start":
        start(gpus, args.record)
    else:
        _write(args.record, {"action": "status", "gpus": sorted(gpus),
                            "workers": _workers()})


if __name__ == "__main__":
    main()
