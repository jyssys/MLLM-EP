"""Read-only live capture of pre-router encoder summaries and router top-k IDs."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INSTALLED = False
_LOCK = threading.Lock()
_SEEN: set[tuple[int, int, int]] = set()


def _control() -> dict[str, Any]:
    path = Path(os.environ["FLASHVEP_PREROUTER_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _append(name: str, row: dict[str, Any]) -> None:
    output = Path(os.environ["FLASHVEP_PREROUTER_RAW"])
    output.mkdir(parents=True, exist_ok=True)
    with _LOCK, (output / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _encoder_features(tensor: torch.Tensor) -> list[float]:
    """Small fixed, permutation-invariant summary; no learned projection."""
    x = tensor.detach().float()
    blocks = list(x.chunk(4, dim=1)) if x.ndim == 2 and x.shape[1] % 4 == 0 else [x]
    features: list[float] = []
    for block in blocks:
        norms = torch.linalg.vector_norm(block, dim=1)
        features.extend([
            float(block.mean()), float(block.std()), float(block.abs().max()),
            float(norms.mean()), float(norms.std()),
        ])
    return features


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    original_capture = RoutedExpertsCapturer.capture
    original_vision = Qwen3_VisionTransformer.forward

    def patched_capture(self: Any, layer_id: int, topk_ids: torch.Tensor) -> None:
        entry = _control()
        metadata = get_forward_context().dp_metadata
        if metadata is not None and entry.get("capture"):
            counts = metadata.num_tokens_across_dp_cpu
            local_tokens = int(counts[self.dp_rank].item())
            tp_rank = int(get_tensor_model_parallel_rank())
            key = (int(entry["wave"]), int(layer_id), tp_rank)
            expected_tokens = int(entry.get("prompt_tokens", 0))
            if local_tokens >= expected_tokens > 0 and key not in _SEEN:
                _SEEN.add(key)
                out = Path(os.environ["FLASHVEP_PREROUTER_RAW"]) / "routes"
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / f"wave{entry['wave']}_dp{self.dp_rank}_tp{tp_rank}_layer{layer_id}.npy",
                        topk_ids.detach().cpu().numpy().astype(np.int16))
        try:
            return original_capture(self, layer_id, topk_ids)
        except AssertionError as error:
            # Preserve the already validated DeepEP dummy-shape workaround only.
            if "unexpected topk_ids batch dim" not in str(error):
                raise
            metadata = get_forward_context().dp_metadata
            if metadata is None:
                raise
            counts = metadata.num_tokens_across_dp_cpu
            local_tokens = int(counts[self.dp_rank].item())
            if local_tokens <= int(topk_ids.shape[0]):
                self._device_buffer[:local_tokens, layer_id, :] = topk_ids[:local_tokens]

    def patched_vision(self: Any, x: torch.Tensor, grid_thw: Any, **kwargs: Any) -> torch.Tensor:
        entry = _control()
        start_ns = time.perf_counter_ns()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        output = original_vision(self, x, grid_thw, **kwargs)
        end.record(torch.cuda.current_stream())
        if entry.get("capture") and get_tensor_model_parallel_rank() == 0:
            end.synchronize()
            grids = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
            merge = int(getattr(self, "spatial_merge_size", 2))
            sizes = [int(t * h * w // (merge * merge)) for t, h, w in grids]
            cursor = 0
            summaries = []
            for size in sizes:
                summaries.append(_encoder_features(output[cursor:cursor + size]))
                cursor += size
            _append(f"encoder.dp{os.environ['VLLM_DP_RANK']}.jsonl", {
                "wave": int(entry["wave"]), "request_id": entry["request_id"],
                "repeat": int(entry["repeat"]), "grid_thw": grids,
                "image_features": summaries, "feature_definition": "four output blocks x [mean,std,max_abs,mean_token_l2,std_token_l2]",
                "vision_cuda_ms": float(start.elapsed_time(end)),
                "vision_host_ms": (time.perf_counter_ns() - start_ns) / 1e6,
            })
        return output

    RoutedExpertsCapturer.capture = patched_capture
    Qwen3_VisionTransformer.forward = patched_vision
