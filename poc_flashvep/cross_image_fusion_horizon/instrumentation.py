"""Causal prefill attention isolation while preserving stock KV-cache writes."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INSTALLED = False
_SAVED_LOGITS: set[str] = set()


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_FUSION_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _layer(name: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


def _manual_attention(module: Any, query: torch.Tensor, key: torch.Tensor,
                      value: torch.Tensor, entry: dict[str, Any],
                      capture_interaction: bool) -> tuple[torch.Tensor, dict[str, float] | None]:
    tokens = query.shape[0]
    query_heads = query.view(tokens, module.num_heads, module.head_size).float()
    key_heads = key.view(tokens, module.num_kv_heads, module.head_size).float()
    value_heads = value.view(tokens, module.num_kv_heads, module.head_size_v).float()
    repeat = module.num_heads // module.num_kv_heads
    if repeat > 1:
        key_heads = key_heads.repeat_interleave(repeat, dim=1)
        value_heads = value_heads.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("thd,shd->hts", query_heads, key_heads)
    scores *= float(module.impl.scale)
    allowed = torch.ones((tokens, tokens), dtype=torch.bool, device=query.device).tril_()
    spans = [tuple(map(int, span)) for span in entry["image_spans"]]

    interaction = None
    if capture_interaction:
        probabilities = torch.softmax(scores.masked_fill(~allowed[None], -torch.inf), dim=-1)

        def mass(query_span: tuple[int, int], key_spans: list[tuple[int, int]]) -> float:
            q0, q1 = query_span
            values = [probabilities[:, q0:q1, k0:k1].sum(dim=-1)
                      for k0, k1 in key_spans]
            return float(torch.stack(values).sum(dim=0).mean().item())

        intra = np.mean([mass(span, [span]) for span in spans])
        cross = mass(spans[1], [spans[0]])
        question = mass((int(entry["post_start"]), tokens), spans)
        interaction = {"intra_image_visual_to_visual": float(intra),
                       "cross_image_visual_to_visual": float(cross),
                       "question_to_visual": float(question)}

    if entry["intervention"] in ("visual", "full"):
        # With causal ordering only image 2 can see image 1.
        second_start, second_end = spans[1]
        first_start, first_end = spans[0]
        allowed[second_start:second_end, first_start:first_end] = False
    if entry["intervention"] == "full":
        post_start = int(entry["post_start"])
        for start, end in spans:
            allowed[post_start:, start:end] = False

    probabilities = torch.softmax(scores.masked_fill(~allowed[None], -torch.inf), dim=-1)
    output = torch.einsum("hts,shd->thd", probabilities, value_heads)
    return output.to(query.dtype).reshape(tokens, -1), interaction


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    original_attention = Attention.forward
    original_logits = LogitsProcessor.forward

    def patched_attention(self: Any, query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor,
                          output_shape: torch.Size | None = None) -> torch.Tensor:
        stock = original_attention(self, query, key, value, output_shape)
        entry = _control()
        layer = _layer(str(self.layer_name))
        if (not entry.get("capture") or query.shape[0] != int(entry.get("prompt_tokens", -1))
                or layer < 0):
            return stock
        capture_interaction = bool(entry.get("interaction_capture"))
        intervene = (entry.get("intervention") in ("visual", "full") and
                     layer < int(entry.get("horizon", 0)))
        if not intervene and not capture_interaction:
            return stock
        custom, interaction = _manual_attention(
            self, query, key, value, entry, capture_interaction)
        if interaction is not None:
            output_dir = Path(os.environ["FLASHVEP_FUSION_INTERACTION"])
            output_dir.mkdir(parents=True, exist_ok=True)
            dp_rank = int(os.environ["VLLM_DP_RANK"])
            tp_rank = int(get_tensor_model_parallel_rank())
            path = output_dir / (f"{entry['capture_id']}.dp{dp_rank}.tp{tp_rank}."
                                 f"layer{layer}.json")
            path.write_text(json.dumps({"layer": layer, **interaction}) + "\n",
                            encoding="utf-8")
        return custom if intervene else stock

    def patched_logits(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor | None:
        logits = original_logits(self, *args, **kwargs)
        entry = _control()
        capture_id = str(entry.get("capture_id", ""))
        if logits is None or not entry.get("capture") or capture_id in _SAVED_LOGITS:
            return logits
        _SAVED_LOGITS.add(capture_id)
        output_dir = Path(os.environ["FLASHVEP_FUSION_LOGITS"])
        output_dir.mkdir(parents=True, exist_ok=True)
        dp_rank = int(os.environ["VLLM_DP_RANK"])
        tp_rank = int(get_tensor_model_parallel_rank())
        path = output_dir / f"{capture_id}.dp{dp_rank}.tp{tp_rank}.npy"
        np.save(path, logits[0].detach().to(torch.float16).cpu().numpy())
        return logits

    Attention.forward = patched_attention
    LogitsProcessor.forward = patched_logits
