"""Make vLLM 0.20 routed-expert capture tolerate DeepEP dummy warmup shapes."""

from __future__ import annotations

import logging
import os
from pathlib import Path


if os.environ.get("FLASHVEP_VISION_TILE_CAPTURE_FIX") == "1":
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.forward_context import get_forward_context

    import numpy as np

    _original_capture = RoutedExpertsCapturer.capture
    _warned = False
    _direct_calls: dict[int, int] = {}

    def _direct_capture(self, layer_id, topk_ids):
        trace_dir = os.environ.get("FLASHVEP_DIRECT_ROUTING_DIR")
        if not trace_dir:
            return
        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return
        counts = metadata.num_tokens_across_dp_cpu
        local_tokens = int(counts[self.dp_rank].item())
        global_tokens = int(counts.sum().item())
        if local_tokens <= 100:
            return
        tp_rank = get_tensor_model_parallel_rank()
        selected = topk_ids
        call = _direct_calls.get(int(layer_id), 0)
        _direct_calls[int(layer_id)] = call + 1
        output = Path(trace_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.save(
            output
            / f"dp{self.dp_rank}_tp{tp_rank}_call{call}_layer{int(layer_id)}.npy",
            selected.detach().to("cpu", non_blocking=False).numpy().astype(np.int16),
        )

    def _capture_or_skip_dummy(self, layer_id, topk_ids):
        global _warned
        _direct_capture(self, layer_id, topk_ids)
        try:
            return _original_capture(self, layer_id, topk_ids)
        except AssertionError as error:
            if "unexpected topk_ids batch dim" not in str(error):
                raise
            context = get_forward_context()
            metadata = context.dp_metadata
            if metadata is not None:
                counts = metadata.num_tokens_across_dp_cpu
                local_tokens = int(counts[self.dp_rank].item())
                global_tokens = int(counts.sum().item())
                rows = int(topk_ids.shape[0])
                if local_tokens <= rows < global_tokens:
                    self._device_buffer[:local_tokens, layer_id, :] = topk_ids[:local_tokens, :]
                    return None
            # DeepEP's startup-only dummy MoE call has fewer rows than the DP
            # padding metadata. It has no request indices and is cleared before
            # serving. Real request shapes continue through the stock capturer.
            if not _warned:
                logging.getLogger(__name__).warning(
                    "Skipping routed-expert capture for DeepEP dummy warmup: %s",
                    error,
                )
                _warned = True
            return None

    RoutedExpertsCapturer.capture = _capture_or_skip_dummy
