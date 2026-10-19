"""Read-only profiler permission screen. No CUDA context or profiler target starts."""
import argparse
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
from poc_ouro_parallelism_deep_discovery.contract import ROOT, write


def command(argv):
    run = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    return dict(argv=argv, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError(target)
    params = Path("/proc/driver/nvidia/params").read_text()
    restriction = int(re.search(r"^RmProfilingAdminOnly:\s*(\d+)", params, re.M)[1])
    process = Path("/proc/self/status").read_text()
    capabilities = int(re.search(r"^CapEff:\s*([0-9a-fA-F]+)", process, re.M)[1], 16)
    ncu = Path("/usr/local/cuda/bin/ncu").resolve(strict=True)
    mode = ncu.stat().st_mode
    version = command([str(ncu), "--version"])
    file_caps = command(["getcap", str(ncu)])
    permitted_credentials = (os.geteuid() == 0 or bool(capabilities & ((1 << 21) | (1 << 38))))
    write(target, dict(timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        euid=os.geteuid(), effective_linux_capabilities_hex=hex(capabilities),
        CAP_SYS_ADMIN=bool(capabilities & (1 << 21)), CAP_PERFMON=bool(capabilities & (1 << 38)),
        observed_RmProfilingAdminOnly=restriction, ncu_path=str(ncu),
        ncu_setuid=bool(mode & 0o4000), ncu_setgid=bool(mode & 0o2000),
        ncu_file_capabilities=file_caps, ncu_version=version,
        driver=command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        source_derived_expectation="Restricted counter profiling is unavailable to these observed credentials."
            if restriction and not permitted_credentials and not file_caps["stdout"].strip() and not mode & 0o4000
            else "UNRESOLVED. No target profiling attempt occurred.",
        measured_error="NOT_RUN. No ERR_NVGPUCTRPERM target trial is claimed.",
        physical_counter_measurement="NOT_RUN", GPU_context_created=False,
        administrative_changes=False, scope="Read-only local metadata plus official permission rule. Not a profiling measurement."))


if __name__ == "__main__":
    main()
