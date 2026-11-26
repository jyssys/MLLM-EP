"""Request-scoped, one-shot descriptor selection for a possible crossing image."""

import json
import math
import os
from pathlib import Path
import re
import time


PREISSUE_ENV = "MLLM_EPD_LATE_PREISSUE_CROSSING_EC"
MAX_FRACTION_ENV = "MLLM_EPD_CROSSING_MAX_SUFFIX_FRACTION"
MAX_TOKENS_ENV = "MLLM_EPD_CROSSING_MAX_SUFFIX_TOKENS"
CROSSING_ROLE = "decode_crossing"
CROSSING_PLAN = "cuda_ipc_crossing_candidate.json"
TORCH_CUDA_IPC_REVISION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


def inline_paths(folder, issued_role):
    return (folder / f"cuda_ipc_inline_{issued_role}.json",
            folder / f"cuda_ipc_inline_{issued_role}_taken.json")


def inline_available(folder, issued_role):
    descriptor, taken = inline_paths(folder, issued_role)
    return descriptor.is_file() and not taken.exists() and not (folder / "cuda_ipc_released.json").exists()


def decode_inline_roles(folder, allow_redirect):
    explicit, _ = inline_paths(folder, "decode")
    if not allow_redirect or explicit.exists() or (folder / "cuda_ipc_consumers.json").exists():
        return ("decode",)
    return ("prefill", CROSSING_ROLE)


def try_mark(path, witness):
    """Create the single owner marker. An existing marker always wins."""
    try:
        with path.open("x") as stream:
            json.dump(witness, stream)
    except FileExistsError:
        return False
    return True


def claim_inline(folder, key, issued_role, worker_role):
    if issued_role == CROSSING_ROLE and worker_role != "decode":
        raise ValueError("A crossing descriptor belongs to the D worker")
    if not inline_available(folder, issued_role):
        return None
    descriptor, taken = inline_paths(folder, issued_role)
    data = json.loads(descriptor.read_text())
    if data["consumer"] != issued_role or [row["key"] for row in data["items"]] != [key]:
        raise ValueError("Unexpected inline descriptor identity")
    witness = dict(consumer_pid=os.getpid(), time=time.time())
    if issued_role != worker_role:
        witness["consumer_role"] = worker_role
    if issued_role == CROSSING_ROLE:
        witness.update(disposition="claimed", token=data["token"])
    return data if try_mark(taken, witness) else None


def claim_crossing_revoke(folder, token):
    """Compete with D for an unused descriptor without opening CUDA storage."""
    if not re.fullmatch("[a-f0-9]{32}", token):
        raise ValueError("Invalid crossing descriptor token")
    _, taken = inline_paths(folder, CROSSING_ROLE)
    return try_mark(taken, dict(disposition="revoked", producer_pid=os.getpid(), token=token, time=time.time()))


def cancellation_marker(root, request_key):
    if not re.fullmatch("[a-f0-9]{64}", request_key):
        raise ValueError("Invalid crossing descriptor request key")
    return Path(root) / "_crossing_cancelled" / (request_key + ".json")


def crossing_candidates(prompt, max_fraction, max_tokens):
    """Select image interiors with at least one integer cut in the suffix envelope."""
    if not math.isfinite(max_fraction) or not 0 <= max_fraction <= .35 or type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("Invalid crossing descriptor suffix envelope")
    length = len(prompt["prompt_token_ids"])
    suffix_bound = max(math.ceil(length * max_fraction), max_tokens)
    minimum_cut = max(1, length - suffix_bound)
    keys = prompt["mm_hashes"]["image"]
    positions = prompt["mm_placeholders"]["image"]
    if len(set(keys)) != len(keys):
        raise ValueError("Crossing descriptors require distinct request-scoped image identifiers")
    selected = []
    for key, position in zip(keys, positions, strict=True):
        left, right = position.offset, position.offset + position.length
        if not 0 <= left < right <= length:
            raise ValueError("Crossing candidate has an invalid original visual span")
        if max(minimum_cut, left + 1) <= min(length - 1, right - 1):
            selected.append((key, left, right))
    return selected
