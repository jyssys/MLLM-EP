"""CPU metadata and read-only GPU inventory. No inference allocation."""

import hashlib
import importlib.metadata as metadata
import platform
import subprocess
import sys
from .common import ROOT, MODEL, FORK, UPSTREAM, write


def cmd(words):
    return subprocess.check_output(words, text=True).strip()


def main():
    source = {}
    for p in sorted((FORK / "vllm_rlt").rglob("*.py")):
        source[str(p.relative_to(FORK))] = hashlib.sha256(p.read_bytes()).hexdigest()
    checkpoint = MODEL / "model.safetensors"
    digest = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    env = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        upstream=UPSTREAM,
        fork=str(FORK),
        fork_HEAD=cmd(["git", "-C", str(FORK), "rev-parse", "HEAD"]),
        shared_runtime="/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages",
        packages={
            n: metadata.version(n)
            for n in (
                "torch",
                "triton",
                "transformers",
                "pytest",
                "numpy",
                "safetensors",
                "flash-attn-4",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
                "matplotlib",
                "ruff",
            )
        },
        checkpoint=str(MODEL),
        checkpoint_sha256=digest,
        source_sha256=source,
        GPUs=cmd(
            [
                "nvidia-smi",
                "-i",
                "4,5,6,7",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv",
            ]
        ),
        topology=cmd(["nvidia-smi", "topo", "-m"]),
        scope="Single-GPU experiments on physical GPU4 only. GPUs5–7 retain burners. Paused PP study untouched.",
        baseline_tests="33 CPU split-prefill tests passed before patch.",
        patched_tests="800 passed, 34 skipped, 322 GPU deselected. Eight separate grouped CPU tests passed.",
        backend="Existing RLT Triton and FlashAttention4 paged paths. No new attention kernel. Static R4 throughout.",
        spec_sha256=hashlib.sha256(
            (ROOT.parent / "ouro_chunk_loop_prefill_joint_poc_spec.md").read_bytes()
        ).hexdigest(),
    )
    write(ROOT / "environment.json", env)


if __name__ == "__main__":
    main()
