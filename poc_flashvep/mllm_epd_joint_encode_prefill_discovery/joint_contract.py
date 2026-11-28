"""Exact image-complete partial ownership on the original Qwen2.5-VL prompt."""

import hashlib
import os
import struct

JOINT_KEY = "campaign_joint_frontier_v1"
VISION_END = 151653


def token_digest(ids):
    """Use the same little-endian int64 token fingerprint as the frozen corpus."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def local_image_prefix():
    return "joint_local_" + os.environ["MLLM_EPD_RUN_NONCE"] + "_"


def plan_from_request(request):
    sampling = request.sampling_params
    if sampling is None:
        return None, {}
    transfer = (sampling.extra_args or {}).get("kv_transfer_params") or {}
    return transfer.get(JOINT_KEY), transfer


def apply_contract(request, role):
    plan, transfer = plan_from_request(request)
    if plan is None:
        return request

    fields = {"mode", "cut", "images", "prefix_sha256", "full_prompt_tokens"}
    compact = plan.get("mode") == "local_suffix_compact"
    if compact:
        fields |= {"metadata_key", "full_sha256"}
    if set(plan) != fields:
        raise ValueError("Unrecognized joint execution contract")
    mode, cut, count = plan["mode"], plan["cut"], plan["images"]
    ids = request.prompt_token_ids
    features = request.mm_features or []
    if type(cut) is not int or type(count) is not int or not 0 < cut < len(ids) or not 0 < count <= len(features):
        raise ValueError("Invalid joint prefix dimensions")
    closing = features[count - 1].mm_position
    if closing.offset + closing.length + 1 != cut or ids[cut - 1] != VISION_END:
        raise ValueError("Joint cut must contain the original closing image delimiter")
    if token_digest(ids[:cut]) != plan["prefix_sha256"]:
        raise ValueError("Rerender changed the original causal prefix")
    if any(item.modality != "image" for item in features):
        raise ValueError("Joint execution currently qualifies image-only Qwen prompts")

    if mode in {"local_suffix", "local_suffix_compact"}:
        if (role != "decode" or len(ids) != plan["full_prompt_tokens"] or count == len(features)
                or not transfer.get("do_remote_prefill") or transfer.get("remote_num_tokens") != cut
                or not transfer.get("remote_block_ids")
                or features[count].mm_position.offset < cut):
            raise ValueError("D-local suffix requires the exact physically imported P prefix and full original prompt")
        if compact and token_digest(ids) != plan["full_sha256"]:
            raise ValueError("Rebuilt original prompt failed at the EngineCore admission boundary")
        selected = features[count:]
    elif mode in {"local_prefix", "remote_prefix"}:
        if role not in ({"decode"} if mode == "local_prefix" else {"decode", "prefill"}):
            raise ValueError("Worker does not own this prefix")
        keep_full_metadata = role == "prefill" and mode == "remote_prefix" and len(ids) == plan["full_prompt_tokens"]
        if ((count != len(features) and not keep_full_metadata) or request.sampling_params.max_tokens != 1
                or transfer.get("do_remote_prefill")):
            raise ValueError("A prefix probe carries only whole prefix images and an internal discarded sample")
        request.prompt_token_ids = ids[:cut]
        selected = features if mode == "local_prefix" else []
    else:
        raise ValueError("Unknown joint execution mode")

    if selected:
        namespace = local_image_prefix()
        for item in selected:
            if item.data is None or "pixel_values" not in item.data or "image_grid_thw" not in item.data:
                raise ValueError("Local vision requires original processed pixels and image grid")
            item.identifier = namespace + item.identifier
    return request
