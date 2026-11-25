"""Release only verified local train_lm.py workers on the experiment GPUs."""
from contextlib import contextmanager
import ctypes
import fcntl
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from .runtime import PAUSE, nvidia, require, sha

BURNER_DIR = Path("/home1/irteam/lm_pretrain")
BURNER_HASH = "175e626d54f61423bef404318c14ab71b356c3e64fdc316036a19a31adaebb77"
KEEPALIVE = b"/home1/irteam/dsd_work/burner_keepalive.sh"
BURNER_PYTHON = "/home/irteam/ddn/envs/kvp/bin/python"
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long


def _pidfd_call(name, *arguments):
    wrapper = getattr(LIBC, name, None)
    if wrapper is not None:
        return wrapper(*arguments)
    require(platform.machine() == "x86_64", "This runtime needs native pidfd wrappers")
    number = {"pidfd_open": 434, "pidfd_send_signal": 424}[name]
    return LIBC.syscall(ctypes.c_long(number), *arguments)


def _argv(pid):
    return (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")


def _target_processes(gpus):
    reverse = {uuid: gpu for gpu, uuid in gpus.items()}
    return [(reverse[uuid], int(pid)) for uuid, pid in nvidia("--query-compute-apps=gpu_uuid,pid")
            if uuid in reverse]


def _command(gpu):
    return [BURNER_PYTHON, "train_lm.py", "--gpu", str(gpu),
            "--hours", "24", "--util_min", "0.80", "--util_max", "0.95"]


def _verify_worker(gpu, pid):
    proc = Path("/proc") / str(pid)
    argv = _argv(pid)
    require(proc.stat().st_uid == os.getuid() and os.getsid(pid) == pid, f"Unowned GPU process {pid}")
    require((proc / "cwd").resolve() == BURNER_DIR.resolve(), f"Unexpected cwd for PID {pid}")
    require(argv == list(map(os.fsencode, _command(gpu))),
            f"GPU {gpu} PID {pid} is not a verified burner")


def _starting_workers(gpus):
    commands = {tuple(map(os.fsencode, _command(gpu))): gpu for gpu in gpus}
    started = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            gpu = commands.get(tuple(_argv(pid)))
            if gpu is not None:
                _verify_worker(gpu, pid)
                started.append((gpu, pid))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return started


def _stop_burners(gpus):
    require(sha(BURNER_DIR / "train_lm.py") == BURNER_HASH, "Local burner code fingerprint changed")
    open_workers = []
    try:
        for gpu, pid in sorted(set(_target_processes(gpus) + _starting_workers(gpus))):
            fd = _pidfd_call("pidfd_open", pid, 0)
            if fd < 0:
                raise OSError(ctypes.get_errno(), f"pidfd_open failed for {pid}")
            open_workers.append((gpu, pid, fd))
            _verify_worker(gpu, pid)
        for gpu, pid, fd in open_workers:
            if _pidfd_call("pidfd_send_signal", fd, signal.SIGTERM, ctypes.c_void_p(), 0) != 0:
                raise OSError(ctypes.get_errno(), f"SIGTERM failed for PID {pid}")
        deadline = time.monotonic() + 30
        while _target_processes(gpus):
            require(time.monotonic() < deadline, "Experiment GPUs did not become idle")
            time.sleep(.5)
        return [{"gpu": gpu, "pid": pid} for gpu, pid, _ in open_workers]
    finally:
        for _, _, fd in open_workers:
            os.close(fd)


def restore_burners(gpus, state_dir):
    require(sha(BURNER_DIR / "train_lm.py") == BURNER_HASH, "Local burner code fingerprint changed")
    occupied = {gpu for gpu, _ in _target_processes(gpus) + _starting_workers(gpus)}
    env = os.environ.copy()
    for key in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "PYTHONPATH",
                "PYTHONHOME", "LD_PRELOAD"):
        env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for gpu in sorted(set(gpus) - occupied):
        with (state_dir.parent / "logs" / f"restored_burner_gpu{gpu}.log").open("xb") as log:
            subprocess.Popen(_command(gpu), cwd=BURNER_DIR, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


@contextmanager
def experiment_lease(gpus, state_dir):
    """Pause existing keepalive temporarily while non-campaign burners keep running."""
    lock_path = state_dir.parent.parent / ".h200_epd_gpu_group.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with PAUSE.open("x") as marker:
            marker.write(f"{os.getpid()}\n")
            inode = os.fstat(marker.fileno()).st_ino
        try:
            # Catch any worker whose launch was already underway when the marker appeared.
            time.sleep(2)
            yield _stop_burners(gpus)
        finally:
            try:
                restore_burners(gpus, state_dir)
            finally:
                if PAUSE.exists() and PAUSE.stat().st_ino == inode:
                    PAUSE.unlink()


def burner_status():
    """Check the existing keepalive and current per-device burner identities."""
    _, = [pid for pid in Path("/proc").iterdir() if pid.name.isdigit() and _is_keepalive(pid)]
    inventory = {int(gpu): uuid for gpu, uuid in nvidia("--query-gpu=index,uuid")}
    observed = _target_processes(inventory)
    for gpu, pid in observed:
        _verify_worker(gpu, pid)
    return observed


def _is_keepalive(pid):
    try:
        return pid.stat().st_uid == os.getuid() and _argv(int(pid.name)) == [b"/bin/bash", KEEPALIVE]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
