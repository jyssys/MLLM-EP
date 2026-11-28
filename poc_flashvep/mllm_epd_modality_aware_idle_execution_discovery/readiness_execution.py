"""Run a fixed, image-complete causal frontier in the pinned Native EPD workers."""

import json
import os
from pathlib import Path
import re
import time


FRONTIER_KEY = "campaign_ready_frontier_v1"
VISION_END = 151653


def contract_path(metadata, key):
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("Invalid request-scoped frontier identity")
    return Path(metadata) / key / "ready_frontier.json"


def load_contract(metadata, key):
    value = json.loads(contract_path(metadata, key).read_bytes())
    if value["key"] != key or value["role"] not in {"prefill", "decode"}:
        raise ValueError("Invalid frontier ownership")
    if not 0 < value["cut"] < value["prompt_tokens"] or not 0 < value["prefix_images"] < len(value["images"]):
        raise ValueError("Invalid causal prefix dimensions")
    return value


def published(ec, image):
    from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import directory

    folder = directory(ec, image)
    return ((folder / "cuda_ipc_inline_prefill.json").is_file()
            and (folder / "cuda_ipc_ready.json").is_file()
            and not (folder / "cuda_ipc_released.json").exists())


def check_cut(value, token_ids, spans):
    cut, size = value["cut"], value["prefix_images"]
    first, end = spans[size - 1]
    next_start, _ = spans[size]
    if not (first < end and end + 1 == cut <= next_start and token_ids[cut - 1] == VISION_END):
        raise ValueError("Frontier must include the original closing vision delimiter")


def install_renderer():
    from ..mllm_epd_prefill_heavy_token_split import processor_handoff as base

    original = base.verify_online_exports
    role = os.environ["MLLM_EPD_ROLE"]
    metadata = os.environ["MLLM_EPD_METADATA_PATH"]

    def verify(inputs, ec):
        key, _ = base.HANDOFF_CONTEXT.get()
        if not contract_path(metadata, key).exists():
            return original(inputs, ec)
        value = load_contract(metadata, key)
        if value["role"] != role:
            return original(inputs, ec)
        prompt, = inputs
        ids = prompt["mm_hashes"]["image"]
        tokens = prompt["prompt_token_ids"]
        positions = prompt["mm_placeholders"]["image"]
        if ids != value["images"] or len(tokens) != value["prompt_tokens"]:
            raise ValueError("Prepared inputs differ from the original frontier contract")
        check_cut(value, tokens, [(p.offset, p.offset + p.length) for p in positions])
        if not all(published(ec, image) for image in ids[:value["prefix_images"]]):
            raise ValueError("An original inline descriptor for the causal prefix is missing")
        return original([{"mm_hashes": {"image": ids[:value["prefix_images"]]}}], ec)

    base.verify_online_exports = verify


def main():
    from . import server_entry

    if os.environ["MLLM_EPD_ROLE"] in {"prefill", "decode"}:
        install_renderer()
    server_entry.main()


if __name__ == "__main__":
    main()
