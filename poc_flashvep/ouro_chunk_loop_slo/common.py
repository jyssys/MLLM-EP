"""Pinned sources and append-only result locations for the follow-up study."""

import hashlib
import json
import os
from pathlib import Path

from poc_flashvep.ouro_chunk_loop_prefill.common import (
    MODEL as MODEL,
    FORK as FORK,
    UPSTREAM as UPSTREAM,
)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery"
OLD = REPO / "poc_flashvep/reports/ouro_chunk_loop_prefill_joint_discovery"
PYTHON = "/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python"
HISTORY = "c458f6a2a764f667a0da6784bcb2592b36626837"
UUIDS = {
    4: "GPU-6076e2f2-5b63-3761-5586-56ceb7df8139",
    5: "GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730",
    6: "GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28",
    7: "GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b",
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def guard():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert value in ("4", "5", "6", "7"), value
    return int(value)


def config(mode, chunk, width=8, blocks=18432, token_budget=1024, graphs=True):
    from poc_flashvep.ouro_chunk_loop_prefill.common import configuration

    c = configuration(mode, chunk, width, graphs=graphs, blocks=blocks)
    from dataclasses import replace

    c["scheduler_config"] = replace(
        c["scheduler_config"], max_num_batched_tokens=token_budget
    )
    c["attention_backend"] = "flash_attn_4"
    return c
