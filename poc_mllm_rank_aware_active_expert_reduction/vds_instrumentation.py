"""Bounded decode attention-to-visual-KV diagnostic for the Q2 gate.

The hook observes the installed FlashAttention implementation without changing
its return value. It is diagnostic, deliberately synchronized, and never used
in clean timing. Only frozen layer/step points are captured.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

_INSTALLED = False
_SEEN: set[tuple[int, int, int, int, int]] = set()


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_RANK_VDS_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _layer(value: Any) -> int:
    name = str(getattr(value, "layer_name", ""))
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


def _emit(payload: dict[str, Any]) -> None:
    root = Path(os.environ["MLLM_RANK_VDS_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"worker_pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    original = FlashAttentionImpl.forward

    def patched(self: Any, layer_module: Any, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: Any,
                output: torch.Tensor, output_scale: torch.Tensor | None = None,
                output_block_scale: torch.Tensor | None = None) -> torch.Tensor:
        entry = _control()
        layer_index = _layer(layer_module)
        by_dp = entry.get("by_dp", {})
        dp_rank = int(os.environ.get("VLLM_DP_RANK", -1))
        request = by_dp.get(str(dp_rank), {})
        layers = set(map(int, entry.get("layers", [])))
        steps = set(map(int, entry.get("steps", [])))
        if (entry.get("capture") and request and layer_index in layers
                and attn_metadata is not None and int(attn_metadata.num_actual_tokens) == 1):
            seq_len = int(attn_metadata.seq_lens[0].item())
            prompt_len = int(request["prompt_token_count"])
            # vLLM decode query for output token index d has cached sequence
            # length prompt_len+d. Therefore d is the 1-based routed decode step.
            decode_step = seq_len - prompt_len
            tp_rank = int(get_tensor_model_parallel_rank())
            seen_key = (int(entry["wave"]), dp_rank, tp_rank, layer_index, decode_step)
            if decode_step in steps and seen_key not in _SEEN:
                _SEEN.add(seen_key)
                q = query[:1].view(1, self.num_heads, self.head_size).float()[0]
                key_cache, value_cache = kv_cache.unbind(0)
                block_size = int(key_cache.shape[1])
                block_count = (seq_len + block_size - 1) // block_size
                block_ids = attn_metadata.block_table[0, :block_count].to(torch.long)
                keys = key_cache.index_select(0, block_ids).reshape(
                    -1, self.num_kv_heads, self.head_size)[:seq_len].float()
                values = value_cache.index_select(0, block_ids).reshape(
                    -1, self.num_kv_heads, self.head_size)[:seq_len].float()
                repeat = self.num_heads // self.num_kv_heads
                if repeat > 1:
                    keys = keys.repeat_interleave(repeat, dim=1)
                    values = values.repeat_interleave(repeat, dim=1)
                scores = torch.einsum("hd,shd->hs", q, keys) * float(self.scale)
                probabilities = torch.softmax(scores, dim=-1)
                visual_start, visual_end = map(int, request["visual_span"])
                visual_prob = probabilities[:, visual_start:visual_end]
                mass = visual_prob.sum(dim=-1)
                visual_contribution = torch.einsum(
                    "hs,shd->hd", visual_prob,
                    values[visual_start:visual_end])
                total_contribution = torch.einsum("hs,shd->hd", probabilities, values)
                ratio = visual_contribution.norm(dim=-1) / total_contribution.norm(dim=-1).clamp_min(1e-12)
                top_count = max(1, self.num_heads // 4)
                _emit({
                    "evidence": "MEASURED_CURRENT_DECODE_FLASHATTENTION_QK_VDS_DIAGNOSTIC",
                    "wave": int(entry["wave"]), "request_id": request["request_id"],
                    "domain": request["domain"], "dp_rank": dp_rank, "tp_rank": tp_rank,
                    "layer": layer_index, "decode_step": decode_step, "seq_len": seq_len,
                    "prompt_token_count": prompt_len, "visual_span": [visual_start, visual_end],
                    "vds_attn_mean": float(mass.mean().item()),
                    "vds_attn_max": float(mass.max().item()),
                    "vds_attn_top_quartile_mean": float(mass.topk(top_count).values.mean().item()),
                    "vds_contribution_mean": float(ratio.mean().item()),
                    "vds_contribution_max": float(ratio.max().item()),
                    "primary_vds_frozen": "vds_attn_mean",
                    "diagnostic_only_not_clean_timing": True,
                })
        return original(self, layer_module, query, key, value, kv_cache,
                        attn_metadata, output, output_scale, output_block_scale)

    FlashAttentionImpl.forward = patched
