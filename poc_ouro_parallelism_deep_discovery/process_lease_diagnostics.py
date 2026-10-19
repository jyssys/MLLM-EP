"""Allowlisted process identity diagnostics. These records authorize no process action."""
import os
from pathlib import Path


def snapshot(pid, expected, role):
    permitted = {"LOOPED_KV_OUTPUT", "CUDA_VISIBLE_DEVICES", "VLLM_WORKER_MULTIPROC_METHOD"}
    if set(expected) != permitted:
        raise ValueError("Identity diagnostics require exactly the three public guard keys")
    process = Path("/proc") / str(pid)
    row = dict(pid=pid, role=role, expected_keys=sorted(expected),
               self_record=pid == os.getpid(), diagnostic_only=True)
    if process.stat().st_uid != os.getuid():
        return dict(row, refused="Not owned by the current UID")
    row.update(uid=process.stat().st_uid, pgid=os.getpgid(pid), sid=os.getsid(pid))
    status = process.joinpath("status").read_text().splitlines()
    row["ppid"] = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
    row["start_time_ticks"] = int(process.joinpath("stat").read_text().rpartition(")")[2].split()[19])
    initial = process.joinpath("environ").read_bytes().split(b"\0")
    row["initial_proc_environment"] = {
        key: dict(present=any(part.startswith((key + "=").encode()) for part in initial),
                  matches=(key + "=" + value).encode() in initial)
        for key, value in expected.items()}
    if pid == os.getpid():
        row["logical_environment"] = {
            key: dict(present=key in os.environ, matches=os.environ.get(key) == value)
            for key, value in expected.items()}
    return row


def native_worker_snapshot(worker, expected):
    """Native collective_rpc supplies the actual rank wrapper as its first argument."""
    return snapshot(os.getpid(), expected, "ACTUAL_TP_RANK_RPC_SELF")
