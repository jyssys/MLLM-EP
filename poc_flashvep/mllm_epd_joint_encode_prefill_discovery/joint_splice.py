"""Compose a current E prefix with locally processed raw suffix images."""

import asyncio
from dataclasses import replace
import json
import os
import time

from .joint_contract import JOINT_KEY, VISION_END, token_digest


VISION_START = 151652


def splice(prefix_ids, prefix_ranges, suffix_ids, suffix_ranges, plan):
    """Validate the exact full sequence and shift only suffix-relative positions."""
    cut = plan["cut"]
    if not 0 < cut < len(prefix_ids) or len(prefix_ranges) != plan["images"] or not suffix_ranges:
        raise ValueError("Partial prompt splice is missing the current E prefix or D suffix")
    final_prefix = prefix_ranges[-1]
    if final_prefix.offset + final_prefix.length + 1 != cut:
        raise ValueError("Live E prefix has different ordered image spans")
    if prefix_ids[cut - 1] != VISION_END or token_digest(prefix_ids[:cut]) != plan["prefix_sha256"]:
        raise ValueError("Live E prefix has the wrong original token sequence")
    start = suffix_ranges[0].offset - 1
    if not 0 <= start < len(suffix_ids) or suffix_ids[start] != VISION_START:
        raise ValueError("Suffix does not start at the original image delimiter")
    full = prefix_ids[:cut] + suffix_ids[start:]
    if len(full) != plan["full_prompt_tokens"] or token_digest(full) != plan["full_sha256"]:
        raise ValueError("Composing the two live media renders changed the full original prompt")
    shift = cut - start
    return full, prefix_ranges + [replace(span, offset=span.offset + shift) for span in suffix_ranges]


def compose(store, current, plan):
    from vllm.multimodal.inputs import MultiModalKwargsItems

    metadata_key = plan["metadata_key"]
    folder = store.directory(metadata_key)
    manifest_path = folder / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_uid != os.getuid():
        raise ValueError("E prefix metadata must belong to the current campaign")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["key"] != metadata_key or manifest["mode"] != "grid_only":
        raise ValueError("Missing original grid-only E prefix render")
    prefix = store.decode_bundle(folder, manifest)
    suffix = current
    if set(prefix.mm_kwargs) != {"image"} or set(suffix["mm_kwargs"]) != {"image"}:
        raise ValueError("The compact joint suffix contract accepts image-only media")
    first = list(prefix.mm_kwargs["image"])
    last = list(suffix["mm_kwargs"]["image"])
    if not all(set(item) == {"image_grid_thw"} for item in first):
        raise ValueError("Unexpected E prefix image representation")
    if not all(set(item) == {"pixel_values", "image_grid_thw"} for item in last):
        raise ValueError("D must receive actual raw suffix pixels and live grid metadata")
    ids, ranges = splice(prefix.prompt["prompt_token_ids"], prefix.mm_placeholders["image"],
                         suffix["prompt_token_ids"], list(suffix["mm_placeholders"]["image"]), plan)
    hashes = prefix.prompt["mm_hashes"]["image"] + [
        "joint_d_raw_" + value for value in suffix["mm_hashes"]["image"]
    ]
    if len(hashes) != len(first) + len(last) or len(set(hashes)) != len(hashes):
        raise ValueError("Prefix and suffix media ownership overlap")
    return dict(suffix, prompt_token_ids=ids, mm_hashes={"image": hashes},
                mm_placeholders={"image": ranges}, mm_kwargs=MultiModalKwargsItems(image=first + last))


def install():
    from vllm.renderers.online_renderer import OnlineRenderer
    from ..mllm_epd_prefill_heavy_token_split.common import ROOT, append
    from ..mllm_epd_prefill_heavy_token_split.processor_handoff import RenderStore

    if os.environ["MLLM_EPD_ROLE"] != "decode":
        return
    original = OnlineRenderer.render_chat
    store = RenderStore(os.environ["MLLM_EPD_METADATA_PATH"])

    async def render(renderer, request, *args, **kwargs):
        plan = (request.kv_transfer_params or {}).get(JOINT_KEY, {})
        if plan.get("mode") != "local_suffix_compact":
            return await original(renderer, request, *args, **kwargs)
        submitted = time.monotonic_ns()
        conversation, inputs = await original(renderer, request, *args, **kwargs)
        if len(inputs) != 1 or inputs[0]["type"] != "multimodal":
            raise ValueError("Expected one raw suffix prompt")
        rendered = time.monotonic_ns()
        merged = await asyncio.to_thread(compose, store, inputs[0], plan)
        done = time.monotonic_ns()
        append(ROOT / "raw/joint_compact_renderer.jsonl", {
            "metadata_key": plan["metadata_key"], "raw_suffix_images": len(inputs[0]["mm_kwargs"]["image"]),
            "grid_only_prefix_images": plan["images"],
            "raw_suffix_render_ms": (rendered - submitted) / 1e6, "prefix_grid_splice_ms": (done - rendered) / 1e6,
        })
        return conversation, [merged]

    OnlineRenderer.render_chat = render
