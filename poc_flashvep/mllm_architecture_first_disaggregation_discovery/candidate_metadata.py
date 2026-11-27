"""Publish architecture counts from public checkpoint indices and headers."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from math import prod
from pathlib import Path
import struct
import urllib.request


PINS = {
    "cog": ("zai-org/cogvlm2-llama3-chat-19B", "f592f291cf528389b2e4776b1e84ecdf6d71fbe3"),
    "idefics": ("HuggingFaceM4/idefics-9b-instruct", "7065afb8a5bb03015700b315f6751dc3dd4e16eb"),
    "ernie": ("baidu/ERNIE-4.5-VL-28B-A3B-Thinking", "e7487b6ff5e726e21bb8bcf9af83b34a82fcdaae"),
    "composer": ("internlm/internlm-xcomposer2-4khd-7b", "a2c222ebd3a723c3dff00232e4f5cc6429f472d1"),
    "ocr2": ("deepseek-ai/DeepSeek-OCR-2", "aaa02f3811945a91062062994c5c4a3f4c0af2b0"),
    "vl2": ("deepseek-ai/deepseek-vl2-tiny", "66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa"),
    "q3moe": ("Qwen/Qwen3-VL-30B-A3B-Instruct", "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"),
    "ar": ("rhymes-ai/Aria", "53a09c77895d7025efa6239ea71cf1a8991064e8"),
    "uni": ("HIT-TMG/Uni-MoE-2.0-Base", "280dff4a7af269450ac619399c1bf046aebf4b76"),
    "bagel": ("ByteDance-Seed/BAGEL-7B-MoT", "5019f57d168e5816e8f3f701b17cc816bb7cf24b"),
    "dsvl1": ("deepseek-ai/deepseek-vl-7b-chat", "6f16f00805f45b5249f709ce21820122eeb43556"),
    "molmo": ("allenai/Molmo2-4B", "042abfa7a38879a376cec03d949eff0aefaa0600"),
    "janus": ("deepseek-ai/Janus-Pro-7B", "5c3eb3fb2a3b61094328465ba61fcd4272090d67"),
    "omni": ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"),
}
PICKLE_INDEX = {"composer", "janus"}
CATEGORY = {
    "cog": ("model.vision.",),
    "idefics": ("model.vision_model.", "model.perceiver_resampler."),
    "ernie": ("vision_model.", "model.resampler_model."),
    "ocr2": ("model.sam_model.", "model.qwen2_model.", "model.projector.", "model.view_seperator"),
    "vl2": ("vision.", "projector.", "view_seperator", "image_newline"),
    "q3moe": ("model.visual.",),
    "ar": ("vision_tower.", "multi_modal_projector."),
    "uni": ("vision_tower.", "vision_aligner."),
    "bagel": ("vit_model.", "vit_pos_embed.", "connector."),
    "dsvl1": ("vision_model.", "aligner."),
    "molmo": ("model.vision_backbone.",),
    "omni": ("thinker.visual.",),
}
REMOTE_EXPERT = {
    "ocr2": (6, 64), "vl2": (6, 64), "q3moe": (8, 128), "ar": (6, 64), "omni": (8, 128),
}


def resource(checkpoint, revision, filename):
    return f"https://huggingface.co/{checkpoint}/resolve/{revision}/{filename}"


def download_metadata(url):
    with urllib.request.urlopen(url, timeout=15) as stream:
        blob = stream.read(4_000_001)
    if len(blob) > 4_000_000:
        raise ValueError("Expected public metadata or source text under 4 MB")
    return blob


def header_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=15) as stream:
        if stream.status != 206 or not stream.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError("Server must honor the exact metadata byte range")
        blob = stream.read(end - start + 1)
    if len(blob) != end - start + 1:
        raise ValueError("Truncated header metadata")
    return blob


def read_header(url):
    size, = struct.unpack("<Q", header_range(url, 0, 7))
    if not 0 < size <= 2_000_000:
        raise ValueError("Unexpected safetensors header size")
    tensors = json.loads(header_range(url, 8, 7 + size))
    tensors.pop("__metadata__", None)
    return tensors


def state_category(model, name):
    if name.startswith(CATEGORY[model]):
        return "vision_encoder_and_projection"
    if model == "cog" and ("vision_expert" in name or ".vision_mlp." in name):
        return "visual_decoder_experts"
    if model == "idefics" and name.startswith("model.gated_cross_attn_layers."):
        return "gated_cross_blocks"
    if model == "ernie":
        if ".experts." in name:
            expert = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            return "routed_language_experts" if expert < 64 else "routed_visual_experts"
        if name.endswith(".mlp.gate.weight_1"):
            return "visual_router"
        if name == "lm_head.weight":
            return "checkpoint_duplicate_tied_head"
    if model == "bagel" and "_moe_gen." in name:
        return "image_generation_transformer_bank"
    if model == "bagel" and not name.startswith("language_model."):
        return "image_generation_auxiliary"
    if model == "uni" and name.startswith(("audio_tower.", "audio_aligner.")):
        return "audio_only_state"
    if model == "omni" and name.startswith(("thinker.audio_tower.", "talker.", "code2wav.")):
        return "audio_only_state"
    if ".experts." in name or (model == "uni" and ".dynamic_real_moe." in name):
        return "routed_language_experts"
    if ".shared_expert" in name or (model == "uni" and ".fixed_real_moe." in name):
        return "shared_language_experts"
    return "shared_language_and_io"


def summarise(name, config_blob, index_blob, tensors):
    checkpoint, revision = PINS[name]
    config, index = json.loads(config_blob), json.loads(index_blob)
    language_scope = config.get("thinker_config", config)
    text = language_scope.get("text_config", language_scope.get("language_config", language_scope.get("llm_config", language_scope)))
    output = {
        "checkpoint": checkpoint, "revision": revision,
        "config_sha256": hashlib.sha256(config_blob).hexdigest(),
        "weight_index_sha256": hashlib.sha256(index_blob).hexdigest(),
        "checkpoint_index_bytes": index["metadata"]["total_size"],
        "checkpoint_shards": len(set(index["weight_map"].values())),
        "config_architectures": config.get("architectures") or text.get("architectures"),
        "configured_context_positions": text.get("max_position_embeddings"),
    }
    if tensors is None:
        return output | {"evidence": "INDEX_ONLY", "stored_total_elements": None}
    if set(tensors) != set(index["weight_map"]):
        raise ValueError(f"{name} header names do not match the pinned index")
    parts = defaultdict(Counter)
    for key, state in tensors.items():
        category = state_category(name, key)
        parts[category]["stored_elements"] += prod(state["shape"])
        begin, end = state["data_offsets"]
        parts[category]["stored_bytes"] += end - begin
    total = sum(row["stored_elements"] for row in parts.values())
    stored_bytes = sum(row["stored_bytes"] for row in parts.values())
    output["tensor_data_bytes_from_headers"] = stored_bytes
    output["header_minus_index_bytes"] = stored_bytes - output["checkpoint_index_bytes"]
    routed = parts.get("routed_language_experts", {}).get("stored_elements", 0)
    shared = sum(parts.get(key, {}).get("stored_elements", 0)
                 for key in ("shared_language_and_io", "shared_language_experts"))
    if name in REMOTE_EXPERT:
        k, n = REMOTE_EXPERT[name]
        output["structural_decode_language_elements_per_token"] = shared + routed * k // n
        output["active_scope"] = "Includes resident vocabulary tables/head and non-routed D tensors, plus top-k selection."
    if name == "ernie":
        k, n = config["moe_k"], config["moe_num_experts"][0]
        output["structural_decode_language_elements_per_token"] = shared + routed * k // n
        output["active_scope"] = "Shared D tensors and routed text FFN top-k, excluding the separately stored tied-head copy."
    component_names = sorted(parts)
    output.update(evidence="INDEX_AND_SAFETENSORS_HEADERS", stored_total_elements=total,
                  stored_elements_and_bytes_by_role={key: dict(parts[key]) for key in component_names})
    dtypes = Counter()
    for state in tensors.values():
        dtypes[state["dtype"]] += prod(state["shape"])
    output["stored_dtype_elements"] = dict(dtypes)
    return output


def survey(name, snapshots):
    model, sha = PINS[name]
    index_name = "pytorch_model.bin.index.json" if name in PICKLE_INDEX else "model.safetensors.index.json"
    if snapshots is not None:
        config = (snapshots / f"{name}_config.json").read_bytes()
        index = (snapshots / f"{name}_{index_name}").read_bytes()
        tensors = None if name in PICKLE_INDEX else json.loads((snapshots / f"{name}_hftensor_headers.json").read_text())
        return summarise(name, config, index, tensors)
    config = download_metadata(resource(model, sha, "config.json"))
    index = download_metadata(resource(model, sha, index_name))
    tensors = None
    if name not in PICKLE_INDEX:
        files = sorted(set(json.loads(index)["weight_map"].values()))
        with ThreadPoolExecutor(max_workers=6) as workers:
            headers = workers.map(read_header, (resource(model, sha, file) for file in files))
            tensors = {key: value for header in headers for key, value in header.items()}
    return summarise(name, config, index, tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshots", type=Path, help="Use captured index/config/header JSON snapshots")
    args = parser.parse_args()
    results = {
        "scope": "Public model indices and bounded HTTP 206 JSON safetensors header ranges. No full checkpoints. "
                 "Counts represent stored tensor elements and may include stored buffers. No runtimes measured.",
        "candidates": {name: survey(name, args.snapshots) for name in PINS},
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
