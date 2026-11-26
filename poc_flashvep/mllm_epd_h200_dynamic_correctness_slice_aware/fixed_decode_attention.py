"""A controlled FA3 single-query split-count intervention for BI1 P/D."""

from functools import wraps
import os


USES = {"single_query_fixed": 0, "multi_query_unchanged": 0}


def install(fixed_count):
    from vllm import envs
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder, get_flash_attn_version

    if (os.environ.get("MLLM_EPD_ROLE") not in ("prefill", "decode") or not envs.VLLM_BATCH_INVARIANT
            or get_flash_attn_version() != 3 or fixed_count not in (4, 8)):
        raise ValueError("Fixed single-query splits require a BI1 FlashAttention 3 LLM worker")
    build = FlashAttentionMetadataBuilder.build

    @wraps(build)
    def build_with_fixed_decode_splits(self, *args, **kwargs):
        metadata = build(self, *args, **kwargs)
        if metadata.max_query_len == 1:
            metadata.max_num_splits = fixed_count
            USES["single_query_fixed"] += 1
        else:
            USES["multi_query_unchanged"] += 1
        return metadata

    FlashAttentionMetadataBuilder.build = build_with_fixed_decode_splits
