"""Pairwise full-vocabulary visual counterfactual metrics during forced replay."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

_BASELINE: dict[tuple[str, int], torch.Tensor] = {}
_LOCK = threading.Lock()


def _write(row: dict[str, Any]) -> None:
    root = Path(os.environ["MLLM_P2_COMMIT_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"sensitivity.pid{os.getpid()}.jsonl"
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _metrics(left: torch.Tensor, right: torch.Tensor, target: int) -> dict[str, Any]:
    left_log = torch.log_softmax(left.float(), dim=-1)
    right_log = torch.log_softmax(right.float(), dim=-1)
    left_p, right_p = left_log.exp(), right_log.exp()
    mixture = 0.5 * (left_p + right_p)
    mixture_log = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    jsd = 0.5 * (
        torch.sum(left_p * (left_log - mixture_log))
        + torch.sum(right_p * (right_log - mixture_log))
    )
    left_top = int(torch.argmax(left).item())
    right_top = int(torch.argmax(right).item())
    left_k = set(int(value) for value in torch.topk(left, 10).indices.tolist())
    right_k = set(int(value) for value in torch.topk(right, 10).indices.tolist())
    return {
        "jsd_nats": float(jsd.item()),
        "abs_target_logprob_delta_nats": float(abs(left_log[target] - right_log[target]).item()),
        "signed_target_logprob_delta_nats": float((left_log[target] - right_log[target]).item()),
        "top1_flip": left_top != right_top,
        "original_top1": left_top, "counterfactual_top1": right_top,
        "top10_overlap": len(left_k & right_k) / 10.0,
    }


class CommitmentLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        if "commit_forced_token_ids" not in extra:
            return
        if extra.get("commit_condition") not in ("A", "B"):
            raise ValueError("commit_condition must be A/B")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        from vllm.distributed import get_tensor_model_parallel_rank

        extra = params.extra_args or {}
        forced = extra.get("commit_forced_token_ids")
        if forced is None:
            return None
        forced_ids = tuple(int(value) for value in forced)
        pair_id = str(extra["commit_pair_id"])
        condition = str(extra["commit_condition"])
        positions = set(int(value) for value in extra["commit_positions"])
        split, domain = str(extra["commit_split"]), str(extra["commit_domain"])

        def process(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            zero_step = len(output_ids)
            position = zero_step + 1
            if position in positions and get_tensor_model_parallel_rank() == 0:
                key = (pair_id, position)
                if condition == "A":
                    if key in _BASELINE:
                        raise RuntimeError(f"duplicate baseline logits {key}")
                    _BASELINE[key] = logits.detach().float().clone()
                else:
                    if key not in _BASELINE:
                        raise RuntimeError(f"missing baseline logits {key}")
                    original = _BASELINE.pop(key)
                    values = _metrics(original, logits, forced_ids[zero_step])
                    _write({
                        "pair_id": pair_id, "split": split, "domain": domain,
                        "position": position, "target_token_id": forced_ids[zero_step],
                        **values, "evidence": "MEASURED_FULL_VOCAB_TEACHER_FORCED_TRUE_EP4",
                    })
            if zero_step < len(forced_ids):
                logits.fill_(-float("inf"))
                logits[forced_ids[zero_step]] = 0.0
            return logits

        return process
