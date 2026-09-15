"""Opt-in dynamic acceptance-budget intervention without runtime edits.

Only the stock confidence threshold changes. The original batch_decode method,
model, expert routing and all EP communication remain unchanged. Decision logs
are observer-heavy and are never used for clean E2E timing claims.
"""

import json
import os
from pathlib import Path


_POLICY = os.environ.get("EP_BUDGET_POLICY", "none")
_LOG = os.environ.get("EP_BUDGET_DECISIONS", "")
if _POLICY != "none" or _LOG:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    import dinfer.decoding.parallel_strategy as strategy

    _original_batch_decode = strategy.ThresholdParallelDecoder.batch_decode
    _call = 0

    if _POLICY.startswith("phase:"):
        _phase_thresholds = tuple(float(value) for value in _POLICY.split(":", 1)[1].split(","))
        if len(_phase_thresholds) != 3 or not all(0 < value <= 1 for value in _phase_thresholds):
            raise ValueError("phase policy requires three thresholds in (0,1]")

        @torch.compile(dynamic=True)
        def _phase_transfer_index(
            logits, temperature, mask_index, x, mask_id, threshold,
            rm_mask=True, use_float64=False, **kwargs,
        ):
            noisy = strategy.add_gumbel_noise(logits, temperature=temperature)
            prediction = noisy.argmax(dim=-1)
            probability = F.softmax(logits.to(torch.float64 if use_float64 else torch.float32), dim=-1)
            token_confidence = probability.gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
            if rm_mask:
                mask_index = mask_index & (prediction != mask_id)
            decoded = torch.where(mask_index, prediction, x)
            confidence = torch.where(mask_index, token_confidence, float("-inf"))
            progress = 1 - mask_index.float().sum(dim=1) / mask_index.shape[1]
            early, middle, late = _phase_thresholds
            local_threshold = torch.where(
                progress < 0.25,
                torch.full_like(progress, early),
                torch.where(progress < 0.75, torch.full_like(progress, middle), torch.full_like(progress, late)),
            )
            actual_threshold = torch.minimum(confidence.max(dim=1)[0] - 1e-5, local_threshold).clamp(min=-1000)
            selected = confidence >= actual_threshold.unsqueeze(-1)
            return decoded, selected

        strategy.get_transfer_index_threshold = _phase_transfer_index
    elif _POLICY.startswith("single:"):
        _single_call, _single_threshold = _POLICY.split(":", 1)[1].split(",")
        _single_call = int(_single_call)
        _single_threshold = float(_single_threshold)
        if _single_call < 0 or not 0 < _single_threshold <= 1:
            raise ValueError("single policy requires call_index,threshold")
    elif _POLICY != "none":
        raise ValueError(f"unknown EP_BUDGET_POLICY={_POLICY}")

    def _captured_batch_decode(self, logits, block_start, x, block_length, iter_threshold=None):
        global _call
        measured = os.environ.get("LLADA_REQUEST_ID", "").startswith("measured_")
        effective_threshold = iter_threshold
        if _POLICY.startswith("single:") and measured and _call == _single_call:
            effective_threshold = _single_threshold
        if not _LOG or (dist.is_initialized() and dist.get_rank() != 0):
            result = _original_batch_decode(self, logits, block_start, x, block_length, effective_threshold)
            if measured:
                _call += 1
            return result
        with torch.no_grad():
            batch, length = x.data.shape
            offset = torch.arange(block_length, device=x.data.device).unsqueeze(0) + block_start.unsqueeze(1)
            before = torch.gather(x.data, 1, offset.clamp(max=length - 1))
        result = _original_batch_decode(self, logits, block_start, x, block_length, effective_threshold)
        with torch.no_grad():
            after = torch.gather(x.data, 1, offset.clamp(max=length - 1))
            prediction = logits.argmax(dim=-1)
            confidence = F.softmax(logits.float(), dim=-1).gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
            selected = (before == self.mask_id) & (after != self.mask_id)
            path = Path(_LOG)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                for row in range(batch):
                    values = confidence[row][before[row] == self.mask_id].detach().cpu().tolist()
                    chosen = confidence[row][selected[row]].detach().cpu().tolist()
                    stream.write(json.dumps({
                        "call": _call, "request_id": os.environ.get("LLADA_REQUEST_ID", "unknown"),
                        "row": row, "block_start": int(block_start[row]),
                        "masked_before": int((before[row] == self.mask_id).sum()),
                        "accepted": int(selected[row].sum()),
                        "base_threshold": float(self.threshold if effective_threshold is None else effective_threshold),
                        "policy": _POLICY, "selected_confidence": chosen,
                        "masked_confidence": values,
                    }, separators=(",", ":")) + "\n")
            if measured:
                _call += 1
        return result

    strategy.ThresholdParallelDecoder.batch_decode = _captured_batch_decode
