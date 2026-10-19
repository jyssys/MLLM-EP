"""Record deterministic CPU tests for the unexecuted graph-argument diagnostic."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NEW_FILES = ["cuda_graph_argument_getter.py", "graph_argument_probe.py",
             "scripts/native_graph_argument_probe.py", "scripts/cuda_graph_argument_cpu_validation.py",
             "tests/test_cuda_graph_argument_getter.py", "tests/test_native_graph_argument_probe_cpu.py",
             "tests/cuda_graph_argument_abi.c"]
UNCHANGED_FILES = ["cuda_graph_event_nodes.py", "cuda_graph_typed_edges.py", "graph_debug_probe.py",
    "graph_typed_edge_probe.py", "worker_probe.py", "operand_binding.py", "runtime.py", "contract.py",
    "scripts/native_graph_dag.py", "scripts/storage_binding_probe.py",
    "reports/cuda_graph_argument_feasibility_v1.md", "artifacts/cuda_graph_argument_feasibility_v1.json"]


def hashes(paths):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def no_cuda_mapping():
    text = Path("/proc/self/maps").read_text()
    return not any(name in text for name in ("libcuda.so", "libcudart.so"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target = args.output.resolve()
    if not target.is_relative_to(ROOT / "artifacts") or target.exists():
        raise ValueError("Use a new output file within the campaign artifacts directory.")
    started = datetime.now(timezone.utc)
    unchanged_before = hashes(UNCHANGED_FILES)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(ROOT / "artifacts"))
    command = [sys.executable, "-m", "unittest",
               "poc_ouro_parallelism_deep_discovery.tests.test_cuda_graph_argument_getter",
               "poc_ouro_parallelism_deep_discovery.tests.test_native_graph_argument_probe_cpu",
               "poc_ouro_parallelism_deep_discovery.tests.test_cuda_graph_typed_edges", "-v"]
    before_no_cuda = no_cuda_mapping()
    result = subprocess.run(command, cwd=ROOT.parent, env=environment, text=True, capture_output=True)
    after_no_cuda = no_cuda_mapping()
    unchanged_after = hashes(UNCHANGED_FILES)
    ended = datetime.now(timezone.utc)
    payload = dict(schema_version=1, worker="independent_axes", CPU_validation=True,
        segment_start_UTC=started.isoformat(), segment_end_UTC=ended.isoformat(),
        elapsed_seconds=(ended - started).total_seconds(), command=command, cwd=str(ROOT.parent),
        python=sys.version, interpreter=sys.executable, platform=platform.platform(),
        environment_overrides={key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "TMPDIR")},
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        no_CUDA_library_mapped_in_driver_before=before_no_cuda,
        no_CUDA_library_mapped_in_driver_after=after_no_cuda,
        GPU_execution="NOT_RUN", model_quality="NOT_RUN", live_getter="NOT_RUN",
        kernel_signature_policy="NO_REAL_KERNEL_POLICY_CREATED",
        new_file_hashes=hashes(NEW_FILES), existing_file_hashes_before=unchanged_before,
        existing_file_hashes_after=unchanged_after,
        existing_files_unchanged=unchanged_before == unchanged_after,
        scope="CPU fake-library argument storage and installed-header type checks only. No model or GPU evidence.")
    with target.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(output=str(target), returncode=result.returncode,
                          existing_files_unchanged=payload["existing_files_unchanged"],
                          elapsed_seconds=payload["elapsed_seconds"]), sort_keys=True))
    if result.returncode or not before_no_cuda or not after_no_cuda or unchanged_before != unchanged_after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
