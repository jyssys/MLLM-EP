"""Record reproducible source, job, and allowed-device provenance."""

import datetime
import hashlib
import importlib.metadata
import json
import subprocess

from .common import ROOT, REPO, FORK, MODEL, digest, write
from .history import table


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main():
    jobs = []
    manifests = []
    for path in sorted((ROOT / "jobs").glob("*.json")):
        d = json.loads(path.read_text())
        target = ROOT / "raw" / d["id"]
        state = (
            "COMPLETE"
            if (target / "completed.json").exists()
            else "PRESERVED_FAILURE"
            if (target / "failure.json").exists()
            else "PENDING_OR_RUNNING"
        )
        identity = (
            json.loads((target / "identity.json").read_text())
            if (target / "identity.json").exists()
            else {}
        )
        for mode, chunk in d.get("cells", [d.get("cell", ["NA", d.get("chunk")])]):
            jobs.append(
                dict(
                    job=d["id"],
                    kind=d["kind"],
                    GPU=identity.get("physical_GPU"),
                    UUID=identity.get("UUID"),
                    mode=mode,
                    C=chunk,
                    shape=d.get("shape"),
                    trials=d.get("trials"),
                    pattern=d.get("pattern"),
                    rate=d.get("rate"),
                    status=state,
                    job_sha256=digest(path),
                    evidence="ACTUAL_DISPATCH_MANIFEST_NOT_ESTIMATED_RESULT",
                )
            )
    table("experiment_matrix.csv", jobs)
    for path in sorted(ROOT.glob("*manifest*.json")):
        if path.name in ("run_manifest.json", "raw_archive_manifest.json"):
            continue
        manifests.append(dict(path=str(path.relative_to(REPO)), sha256=digest(path)))
    code = [
        dict(path=str(p.relative_to(REPO)), sha256=digest(p))
        for p in sorted((REPO / "poc_flashvep/ouro_chunk_loop_slo").glob("*"))
        if p.is_file() and p.suffix in (".py", ".sh")
    ]
    versions = {}
    for name in (
        "torch",
        "flash-attn-4",
        "triton",
        "transformers",
        "numpy",
        "matplotlib",
        "pytest",
        "ruff",
    ):
        versions[name] = importlib.metadata.version(name)
    model_path = __import__("pathlib").Path(MODEL)
    model_files = [
        dict(name=p.name, sha256=digest(p), bytes=p.stat().st_size)
        for p in sorted(model_path.glob("*"))
        if p.is_file()
        and (
            p.suffix in (".json", ".safetensors", ".py", ".jinja")
            or p.name.endswith(".model")
        )
    ]
    runtime_patch = command(["git", "diff", "--binary"], FORK)
    reused = [
        dict(path=str(path.relative_to(REPO)), sha256=digest(path))
        for path in sorted((REPO / "poc_flashvep/ouro_chunk_loop_prefill").glob("*.py"))
    ]
    specification = ROOT / "inputs/authoritative_spec.md"
    write(
        ROOT / "run_manifest.json",
        dict(
            recorded_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            evidence="SOURCE_AND_ENVIRONMENT_SNAPSHOT_NOT_PERFORMANCE",
            repo_branch=command(["git", "branch", "--show-current"], REPO),
            parent=command(["git", "rev-parse", "HEAD"], REPO),
            RLT_revision=command(["git", "rev-parse", "HEAD"], FORK),
            RLT_diff_stat=command(["git", "diff", "--stat"], FORK),
            RLT_diff_sha256=hashlib.sha256(runtime_patch.encode()).hexdigest(),
            RLT_diff=runtime_patch,
            reused_experiment_code=reused,
            authoritative_specification=dict(
                path=str(specification.relative_to(REPO)),
                sha256=digest(specification),
            ),
            local_profiler_files=[
                dict(
                    name=path.name,
                    bytes=path.stat().st_size,
                    sha256=digest(path),
                    publication="LOCAL_ONLY. Portable CSV exports are archived separately.",
                )
                for path in sorted((ROOT / "private_profiles").glob("*"))
                if path.is_file() and path.suffix in (".nsys-rep", ".sqlite")
            ],
            model=MODEL,
            model_files=model_files,
            versions=versions,
            GPUs=command(
                [
                    "nvidia-smi",
                    "-i",
                    "4,5,6,7",
                    "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
                    "--format=csv",
                ]
            ),
            nsys=command(["nsys", "--version"]),
            manifests=manifests,
            code=code,
            paused_project="Native-vLLM Ouro Loop PP. Not resumed.",
            old_staged_diff_sha256="7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9",
            GPU_parallelism="Four independent Single-GPU experiments. Not TP, PP or shared DP serving scaling.",
        ),
    )
    print("INVENTORY", len(jobs), "matrix rows", len(manifests), "manifests")


if __name__ == "__main__":
    main()
