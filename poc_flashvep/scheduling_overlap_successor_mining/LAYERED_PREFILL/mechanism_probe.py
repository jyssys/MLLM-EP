"""Opt-in TP layer/group timing and sampled routed-weight traffic proxy.

Primary request timings keep this OFF. Route-copy samples are flagged and are
not clean CUDA cost observations. No cross-device timestamp subtraction.
"""
import functools
import json
import os
from collections import deque
from pathlib import Path
import time

import torch
from nanovllm.utils.context import get_context
from nanovllm.layers.fused_moe import FusedMoE


def install(runner):
    root = Path(os.environ["SUCCESSOR_LP_TRACE"])
    root.mkdir(parents=True, exist_ok=True)
    sink = (root / f"rank{runner.rank}_pid{os.getpid()}.jsonl").open("a", buffering=1)
    pending = deque()
    current = None
    layer_name = None
    sequence = 0
    cap = int(os.environ.get("SUCCESSOR_LP_TRACE_CAP", "2000"))
    route_every = int(os.environ.get("SUCCESSOR_LP_ROUTE_EVERY", "0"))
    sink.write(json.dumps({"kind": "instrumentation", "rank": runner.rank,
                           "world_size": runner.world_size, "topology": "TP_NOT_EP",
                           "visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"],
                           "model": type(runner.model).__name__,
                           "parameter_dtype": str(next(runner.model.parameters()).dtype),
                           "route_every": route_every}) + "\n")

    def flush():
        while pending and pending[0][2].query():
            record, start, end, route = pending.popleft()
            if start is not None:
                record["cuda_ms"] = start.elapsed_time(end)
            if route is not None:
                histogram = torch.bincount(route.flatten().long(), minlength=128).tolist()
                active = sum(x > 0 for x in histogram)
                prefill = record["prefill_tokens"]
                assert 0 <= prefill <= route.shape[0]
                prefill_hist = torch.bincount(route[:prefill].flatten().long(), minlength=128).tolist()
                decode_hist = torch.bincount(route[prefill:].flatten().long(), minlength=128).tolist()
                assert histogram == [a+b for a, b in zip(prefill_hist, decode_hist)]
                pre_only = sum(a > 0 and b == 0 for a, b in zip(prefill_hist, decode_hist))
                cfg = runner.config.hf_config
                record.update(expert_histogram=histogram, active_experts=active,
                              prefill_expert_histogram=prefill_hist,
                              decode_expert_histogram=decode_hist,
                              prefill_active_experts=sum(x > 0 for x in prefill_hist),
                              decode_active_experts=sum(x > 0 for x in decode_hist),
                              experts_needed_only_by_prefill=pre_only,
                              total_assignments=route.numel(),
                              # Compulsory weight bytes only; not measured DRAM.
                              routed_weight_bytes_proxy_per_tp_rank=
                              active * 3 * cfg.hidden_size * cfg.moe_intermediate_size
                              * 2 // runner.world_size)
            sink.write(json.dumps(record) + "\n")

    def timed(function, stage, is_layer=False):
        @functools.wraps(function)
        def call(*args, **kwargs):
            nonlocal layer_name
            if current is None:
                return function(*args, **kwargs)
            previous_layer = layer_name
            if is_layer:
                layer_name = stage
            context = get_context()
            record = {**current, "kind": "cuda_stage", "stage": stage,
                      "prefill_tokens": context.len_prefill,
                      "prefill_compute_layers": sorted(context.prefill_compute_layers or []),
                      "host_start_s": time.perf_counter()}
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            try:
                return function(*args, **kwargs)
            finally:
                end.record()
                record["host_end_s"] = time.perf_counter()
                pending.append((record, start, end, None))
                layer_name = previous_layer
        return call

    runner.run_model = timed(runner.run_model, "full_layer_stack")
    for name, module in runner.model.named_modules():
        if type(module).__name__.endswith("DecoderLayer"):
            module.forward = timed(module.forward, "layer:" + name, True)

    original_select = FusedMoE.select_experts

    def select(*args, **kwargs):
        weights, ids = original_select(*args, **kwargs)
        if current is not None and current["route_sampled"] and runner.rank == 0:
            cpu = torch.empty(ids.shape, dtype=ids.dtype, device="cpu", pin_memory=True)
            cpu.copy_(ids, non_blocking=True)
            done = torch.cuda.Event(enable_timing=False)
            done.record()
            pending.append(({**current, "kind": "route_histogram", "layer": layer_name,
                             "M": ids.shape[0], "top_k": ids.shape[1],
                             # Native scheduler returns prefill seqs before
                             # decode seqs; inactive-layer paths set this to 0.
                             "prefill_tokens": get_context().len_prefill}, None, done, cpu))
        return weights, ids
    FusedMoE.select_experts = staticmethod(select)
    original_run = runner.run

    @functools.wraps(original_run)
    def run(seqs):
        nonlocal current, sequence
        flush()
        sequence += 1
        if sequence <= cap:
            current = {"rank": runner.rank, "local_invocation_id": sequence,
                       "route_sampled": bool(route_every and sequence % route_every == 0),
                       "requests": [{"request_id": s.seq_id, "status": str(s.status),
                                     "stage": s.stage, "num_stages": s.num_stages,
                                     "tokens_to_process": s.num_tokens_to_process,
                                     "processed_tokens": s.num_processed_tokens,
                                     "length": len(s)} for s in seqs]}
        host_start = time.perf_counter()
        try:
            return original_run(seqs)
        finally:
            if current is not None:
                sink.write(json.dumps({**current, "kind": "run_host",
                                       "host_start_s": host_start,
                                       "host_end_s": time.perf_counter()}) + "\n")
            current = None
            flush()
    runner.run = run
