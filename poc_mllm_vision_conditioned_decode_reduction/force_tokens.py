"""Request-local exact token forcing for diagnostic causal replay."""

from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class ForceTokensLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        forced = (sampling_params.extra_args or {}).get("vcdr_forced_token_ids")
        if forced is None:
            return
        if not isinstance(forced, list) or not forced or any(
            not isinstance(value, int) or value < 0 for value in forced
        ):
            raise ValueError("vcdr_forced_token_ids must be non-empty non-negative integers")
        if sampling_params.max_tokens != len(forced):
            raise ValueError("max_tokens must equal forced token count")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        forced = (params.extra_args or {}).get("vcdr_forced_token_ids")
        if forced is None:
            return None
        ids = tuple(int(value) for value in forced)

        def force(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            step = len(output_ids)
            if step < len(ids):
                token = ids[step]
                if token >= logits.shape[-1]:
                    raise RuntimeError(f"forced token {token} exceeds vocab {logits.shape[-1]}")
                logits.fill_(-float("inf"))
                logits[token] = 0.0
            return logits

        return force
