"""Inspect completed v4 jobs. Export host graphs. Do not import Torch."""

import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from validate_storage_graph_provenance_cpu_v1 import export_job


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_job(job):
    command = job["command"]
    size = command[command.index("--size") + 1]
    parallel = int(command[command.index("--tp") + 1])
    mode = command[command.index("--weight-storage-control") + 1]
    base = ROOT / "outputs" / job["name"] / f"{size}_R4_TP{parallel}"
    destination = ROOT / "analysis/storage_provenance_v4" / job["name"]
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        export_job(base / "compile_cache", size, mode, parallel, destination)
    manifest = json.loads(manifest_path.read_text())
    exports_intact = all(
        digest(Path(record["source_path"])) == record["source_sha256"]
        and digest(Path(record["gzip_path"])) == record["gzip_sha256"]
        and hashlib.sha256(gzip.decompress(Path(record["gzip_path"]).read_bytes())).hexdigest()
        == record["source_sha256"] for record in manifest["records"]
    )
    environment_path = base / "cache_environment.json"
    environment = json.loads(environment_path.read_text())
    expected = {name: str(base / suffix) for name, suffix in [
        ("VLLM_CACHE_ROOT", "vllm_cache"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache"),
        ("TRITON_CACHE_DIR", "triton_cache"),
    ]}
    console = ROOT / "outputs" / job["name"] / "console.log"
    path_lines = []
    aot_paths, compile_paths = [], []
    for number, line in enumerate(console.read_text().split("\n"), 1):
        if "AOT compilation from path " in line or "saved AOT compiled function to " in line:
            path = line.split("path ")[-1] if "from path " in line else line.split("function to ")[-1]
            aot_paths.append(Path(path))
            path_lines.append({"line": number, "text": line})
        if "Using cache directory: " in line:
            path = line.split("Using cache directory: ")[-1].split(" for vLLM")[0]
            compile_paths.append(Path(path))
            path_lines.append({"line": number, "text": line})
    aot_root = base / "vllm_cache/torch_compile_cache/torch_aot_compile"
    models = sorted(aot_root.glob("*/rank_*_*/model"))
    inductor_roots = sorted(aot_root.glob("*/inductor_cache"))
    triton_root = base / "triton_cache"
    groups = sorted(triton_root.glob("*/__grp__*.json"))
    child_paths = [Path(value) for path in groups
                   for value in json.loads(path.read_text()).get("child_paths", {}).values()]
    metadata_paths = []
    for path in sorted((base / "compile_cache").glob("rank_*_*/backbone/vllm_compile_cache.py")):
        for value in ast.literal_eval(path.read_text()).values():
            metadata_paths.append(Path(value["graph_handle"][1]))
    checks = {
        "graph_provenance": manifest["status"] == "PASS",
        "gzip_and_source_hashes": exports_intact,
        "environment_set_before_vllm_import": environment.get("before_vllm_import") is True,
        "environment_roots_match_job": environment.get("variables") == expected,
        "console_aot_paths_within_job_root": bool(aot_paths) and all(p.resolve().is_relative_to(aot_root.resolve()) for p in aot_paths),
        "console_compile_paths_within_job_root": bool(compile_paths) and all(p.resolve().is_relative_to((base / "compile_cache").resolve()) for p in compile_paths),
        "aot_model_rank_set": {p.parent.name for p in models} == {f"rank_{rank}_0" for rank in range(parallel)},
        "forced_inductor_roots_exist": bool(inductor_roots) and all(p.is_dir() for p in inductor_roots),
        "triton_groups_exist": bool(groups),
        "triton_children_within_job_and_exist": bool(child_paths) and all(p.resolve().is_relative_to(triton_root.resolve()) and p.exists() for p in child_paths),
        "compiled_piece_paths_within_job_and_exist": bool(metadata_paths) and all(p.resolve().is_relative_to((base / "compile_cache").resolve()) and p.exists() for p in metadata_paths),
    }
    return {
        "job": job["name"], "size": size, "P": parallel, "mode": mode,
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "runtime_pointer_identity": "NOT_CAPTURED_AT_COMPILED_CALL_BOUNDARY",
        "export_manifest": str(manifest_path), "export_manifest_sha256": digest(manifest_path),
        "environment_path": str(environment_path), "environment_sha256": digest(environment_path),
        "console_path": str(console), "console_sha256": digest(console),
        "requested_roots": expected, "console_path_records": path_lines,
        "aot_models": [str(path) for path in models],
        "actual_forced_inductor_roots": [str(path) for path in inductor_roots],
        "initial_inductor_root_exists": (base / "inductor_cache").exists(),
        "triton_group_count": len(groups), "triton_child_path_count": len(child_paths),
        "compiled_piece_path_count": len(metadata_paths),
        "ranks": [{key: record[key] for key in ["rank", "top_weight_arguments", "depth_prefix_counts", "source_sha256", "gzip_sha256"]}
                  for record in manifest["records"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "artifacts/storage_fully_isolated_queue_v4.json")
    args = parser.parse_args()
    queue_payload = args.queue.read_bytes()
    queue = json.loads(queue_payload)
    completed = [job for job in queue["jobs"] if job["status"] == "PASS" and job.get("exit_code") == 0]
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "queue": str(args.queue), "queue_sha256": hashlib.sha256(queue_payload).hexdigest(),
        "queue_status": queue["status"], "jobs_in_snapshot": len(queue["jobs"]),
        "pending_jobs": [job["name"] for job in queue["jobs"] if job not in completed],
        "evidence_class": "CPU inspection of saved graph and cache metadata",
        "runtime_pointer_identity": "NOT_CAPTURED_AT_COMPILED_CALL_BOUNDARY",
        "timing_attribution": "NOT_EVALUATED",
        "records": [inspect_job(job) for job in completed],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
