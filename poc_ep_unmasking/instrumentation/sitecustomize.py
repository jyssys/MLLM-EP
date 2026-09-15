"""Opt-in, observer-heavy unmask-decision capture for the stock dInfer runner.

Only enabled when EP_UNMASK_ATLAS_DIR is set.  This file is placed first on
PYTHONPATH so multiprocessing spawn children install the same wrapper without
changing the shared dInfer checkout.
"""

import json
import os
from pathlib import Path


_ATLAS_DIR = os.environ.get("EP_UNMASK_ATLAS_DIR")
if _ATLAS_DIR:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from dinfer.decoding.parallel_strategy import ThresholdParallelDecoder

    _original_batch_decode = ThresholdParallelDecoder.batch_decode
    _call_index = 0

    def _captured_batch_decode(
        self, logits, block_start, x, block_length, iter_threshold=None
    ):
        global _call_index
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return _original_batch_decode(
                self, logits, block_start, x, block_length, iter_threshold
            )

        with torch.no_grad():
            batch, total_length = x.data.shape
            offsets = (
                torch.arange(block_length, device=x.data.device).unsqueeze(0)
                + block_start.unsqueeze(1)
            )
            before = torch.gather(x.data, 1, offsets.clamp(max=total_length - 1))
            prediction = logits.argmax(dim=-1)
            probability = F.softmax(logits.float(), dim=-1)
            confidence = probability.gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
            eligible = (before == self.mask_id) & (prediction != self.mask_id)

        result = _original_batch_decode(
            self, logits, block_start, x, block_length, iter_threshold
        )

        with torch.no_grad():
            after = torch.gather(x.data, 1, offsets.clamp(max=total_length - 1))
            selected = eligible & (after != self.mask_id)
            confidence_cpu = confidence.detach().cpu()
            eligible_cpu = eligible.detach().cpu()
            selected_cpu = selected.detach().cpu()
            before_cpu = before.detach().cpu()
            latest_ep_invocation = None
            latest_ep_layer = None
            ep_trace_dir = os.environ.get("LLADA_EP_TRACE_DIR")
            if ep_trace_dir:
                ep_rank0 = Path(ep_trace_dir) / "ep_rank0.jsonl"
                if ep_rank0.exists():
                    with ep_rank0.open("rb") as ep_stream:
                        ep_stream.seek(0, 2)
                        ep_stream.seek(max(0, ep_stream.tell() - 131072))
                        last_line = ep_stream.read().splitlines()[-1]
                    latest_ep = json.loads(last_line)
                    latest_ep_invocation = latest_ep.get("invocation")
                    latest_ep_layer = latest_ep.get("layer")
            record_common = {
                "request_id": os.environ.get("LLADA_REQUEST_ID", "unknown"),
                "call_index": _call_index,
                "sequence_ids": json.loads(os.environ.get("LLADA_TRACE_SEQ_IDS", "[]")),
                "block_starts": block_start.detach().cpu().tolist(),
                "threshold": float(self.threshold if iter_threshold is None else iter_threshold),
                "block_length": int(block_length),
                "ep_latest_invocation": latest_ep_invocation,
                "ep_latest_layer": latest_ep_layer,
            }
            atlas_dir = Path(_ATLAS_DIR)
            atlas_dir.mkdir(parents=True, exist_ok=True)
            with (atlas_dir / "unmask_rank0.jsonl").open("a") as stream:
                for row in range(batch):
                    eligible_positions = torch.where(eligible_cpu[row])[0].tolist()
                    selected_positions = torch.where(selected_cpu[row])[0].tolist()
                    selected_values = [float(confidence_cpu[row, p]) for p in selected_positions]
                    unselected_values = [
                        float(confidence_cpu[row, p])
                        for p in eligible_positions
                        if p not in selected_positions
                    ]
                    stream.write(
                        json.dumps(
                            {
                                **record_common,
                                "batch_row": row,
                                "remaining_before": int((before_cpu[row] == self.mask_id).sum()),
                                "masked_positions": torch.where(before_cpu[row] == self.mask_id)[0].tolist(),
                                "selected_positions": selected_positions,
                                "eligible_positions": eligible_positions,
                                "confidences": [
                                    float(confidence_cpu[row, p]) for p in eligible_positions
                                ],
                                "selected_cutoff": min(selected_values) if selected_values else None,
                                "best_unselected": max(unselected_values) if unselected_values else None,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        _call_index += 1
        return result

    ThresholdParallelDecoder.batch_decode = _captured_batch_decode
