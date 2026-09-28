"""Exact per-request token forcing for causal teacher-forced vLLM replay."""
from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class ForceTokensLogitsProcessor(AdapterLogitsProcessor):
    """Force token ``i`` at step ``i`` using request-local ``extra_args``.

    This changes only token selection. The full model still executes causally
    at every step and computes routing from the current image-conditioned KV
    state. It is deliberately registered only for controlled replay runs.
    """

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        forced = extra.get("mllm_vds_forced_token_ids")
        if forced is None:
            return
        if not isinstance(forced, list) or not forced:
            raise ValueError("mllm_vds_forced_token_ids must be a non-empty list")
        if any(not isinstance(value, int) or value < 0 for value in forced):
            raise ValueError("forced token ids must be non-negative integers")
        if sampling_params.max_tokens != len(forced):
            raise ValueError("max_tokens must equal forced token count")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        extra: dict[str, Any] = params.extra_args or {}
        forced = extra.get("mllm_vds_forced_token_ids")
        if forced is None:
            return None
        forced_ids = tuple(int(value) for value in forced)

        def force(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            step = len(output_ids)
            if step >= len(forced_ids):
                return logits
            token = forced_ids[step]
            if token >= logits.shape[-1]:
                raise RuntimeError(f"forced token {token} exceeds vocab {logits.shape[-1]}")
            logits.fill_(-float("inf"))
            logits[token] = 0.0
            return logits

        return force
