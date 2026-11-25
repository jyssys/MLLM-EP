"""Read-only host inventory and between-cohort telemetry."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def snapshot(*, topology=False):
    fields = ("index,uuid,pstate,memory.total,memory.used,utilization.gpu,utilization.memory,"
              "temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory")
    counters = [line for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith(("cpu ", "ctxt ", "procs_running ", "procs_blocked "))]
    value = {
        "time_unix": time.time(), "time_monotonic": time.monotonic(),
        "load_average": os.getloadavg(), "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_counters": counters, "memory": Path("/proc/meminfo").read_text(),
        "gpu": command(["nvidia-smi", f"--query-gpu={fields}", "--format=csv"]),
    }
    if topology:
        value.update(topology=command(["nvidia-smi", "topo", "-m"]),
                     driver=command(["nvidia-smi"]), cpu=command(["lscpu"]),
                     disk=command(["df", "-h", str(Path.cwd())]))
        if shutil.which("numactl"):
            value["numa"] = command(["numactl", "--hardware"])
    return value


@contextmanager
def sample_nvml(path, device_by_role, interval_seconds=.5):
    """Sample hardware counters off the request loop and write after drain."""
    import pynvml as nvml

    stop = threading.Event()
    rows = []
    errors = []

    def collect():
        try:
            nvml.nvmlInit()
            try:
                devices = {role: nvml.nvmlDeviceGetHandleByIndex(index) for role, index in device_by_role.items()}
                while not stop.is_set():
                    begin = time.monotonic()
                    reading = {}
                    for role, handle in devices.items():
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        reading[role] = {
                            "gpu_utilization_percent": utilization.gpu,
                            "memory_utilization_percent": utilization.memory,
                            "memory_used_bytes": nvml.nvmlDeviceGetMemoryInfo(handle).used,
                            "temperature_c": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                            "power_w": nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                            "sm_clock_mhz": nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
                        }
                    rows.append({"monotonic": begin, "sample_seconds": time.monotonic() - begin, "roles": reading})
                    stop.wait(max(0, interval_seconds - (time.monotonic() - begin)))
            finally:
                nvml.nvmlShutdown()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")

    thread = threading.Thread(target=collect, name="wide-nvml", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
        path.write_text(json.dumps({"scope": "NVML sample-window utilization and device memory. GPU busy requires CUPTI.",
                                    "interval_seconds": interval_seconds, "rows": rows, "errors": errors,
                                    "sampler_drained": not thread.is_alive()}) + "\n")
