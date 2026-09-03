"""Exact-route DeepEP dispatch/combine replay for Nsight fallback evidence.

This is not a synthetic route: the top-k layout is loaded from the existing
Qwen3-VL layer-24 capture.  It deliberately omits expert GEMM because this
bounded fallback has no model-weight loader; expert compute is source-inferred
and cross-referenced to the prior real serving run in the report.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

CAPTURE = Path("/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322/layer24_capture.pt")
OUT = Path(os.environ["FLASHVEP_ATLAS_RESULT_DIR"])


def stats(values: list[float]) -> dict[str, float]:
    q = sorted(values)
    return {"median_ms": float(statistics.median(q)),
            "p95_ms": float(q[max(0, int(0.95 * len(q)) - 1)]),
            "mean_ms": float(statistics.fmean(q)),
            "cv": float(statistics.pstdev(q) / statistics.fmean(q)) if statistics.fmean(q) else 0.0,
            "n": len(q)}


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"expected EP4 replay, got WORLD_SIZE={world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    import deep_ep

    capture_bytes = CAPTURE.read_bytes()
    capture = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    hidden = capture["post_attention_hidden"].to(device=device, dtype=torch.bfloat16).contiguous()
    ids = capture["topk_expert_ids"].to(device=device, dtype=deep_ep.topk_idx_t).contiguous()
    weights = capture["topk_weights"].to(device=device).contiguous()
    if hidden.shape[0] != ids.shape[0] or ids.shape[1] != 8:
        raise RuntimeError(f"capture shape mismatch hidden={tuple(hidden.shape)} ids={tuple(ids.shape)}")

    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(
        dist.group.WORLD, 256 * 1024 * 1024, 0,
        low_latency_mode=False, num_qps_per_rank=1, explicitly_destroy=True,
    )
    dist.barrier()
    warmups, iterations = 3, 20
    rows: list[dict[str, float | int]] = []
    for it in range(warmups + iterations):
        dist.barrier()
        torch.cuda.synchronize(device)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
        e4 = torch.cuda.Event(enable_timing=True); e5 = torch.cuda.Event(enable_timing=True)
        h0 = time.perf_counter_ns(); e0.record()
        torch.cuda.nvtx.range_push("DEEPEP_LAYOUT")
        layout = buffer.get_dispatch_layout(ids, 128, async_finish=False, allocate_on_comm_stream=False)
        torch.cuda.nvtx.range_pop(); e1.record(); e2.record()
        npr, nrr, nep, in_rank, prev = layout
        torch.cuda.nvtx.range_push("DEEPEP_DISPATCH")
        recv_hidden, recv_ids, recv_weights, recv_counts, handle, dispatch_event = buffer.dispatch(
            x=hidden, handle=None, num_tokens_per_rank=npr,
            num_tokens_per_rdma_rank=nrr, is_token_in_rank=in_rank,
            num_tokens_per_expert=nep, topk_idx=ids, topk_weights=weights,
            expert_alignment=1, config=deep_ep.Buffer.get_dispatch_config(world),
            previous_event=prev, async_finish=True, allocate_on_comm_stream=False,
        )
        dispatch_event.current_stream_wait(); torch.cuda.nvtx.range_pop(); e3.record(); e4.record()
        torch.cuda.nvtx.range_push("DEEPEP_COMBINE")
        combined, _, combine_event = buffer.combine(
            x=recv_hidden, handle=handle, topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world), async_finish=True,
            allocate_on_comm_stream=False,
        )
        combine_event.current_stream_wait(); torch.cuda.nvtx.range_pop(); e5.record(); e5.synchronize()
        h1 = time.perf_counter_ns()
        if it >= warmups:
            rows.append({"iteration": it - warmups, "rank": rank,
                         "physical_gpu": [1, 2, 3, 4][local_rank],
                         "token_rows": int(hidden.shape[0]),
                         "assignments": int(hidden.shape[0] * 8),
                         "received_rows": int(recv_hidden.shape[0]),
                         "layout_ms": float(e0.elapsed_time(e1)),
                         "dispatch_ms": float(e2.elapsed_time(e3)),
                         "combine_ms": float(e4.elapsed_time(e5)),
                         "total_ms": float(e0.elapsed_time(e5)),
                         "host_ms": (h1 - h0) / 1e6})
        del layout, recv_hidden, recv_ids, recv_weights, recv_counts, handle, dispatch_event, combine_event, combined
        dist.barrier()

    payload = {
        "rank": rank, "local_rank": local_rank,
        "physical_gpu": [1, 2, 3, 4][local_rank],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "capture_path": str(CAPTURE),
        "capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "hidden_shape": list(hidden.shape), "topk_shape": list(ids.shape),
        "dtype": str(hidden.dtype), "warmups": warmups, "iterations": iterations,
        "deep_ep_module": str(Path(deep_ep.__file__).resolve()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "rows": rows,
        "stats": {k: stats([float(r[k]) for r in rows]) for k in ("layout_ms", "dispatch_ms", "combine_ms", "total_ms", "host_ms")},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"replay_rank{rank}.json").write_text(json.dumps(payload, indent=2) + "\n")
    buffer.destroy()
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
