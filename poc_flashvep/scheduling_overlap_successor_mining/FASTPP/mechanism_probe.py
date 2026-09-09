"""Opt-in same-device stage/layer timing with explicit request association.

This is a measurement hook, not a scheduling change. Run separately from primary
uninstrumented E2E runs. Never synchronizes CUDA in the measured path. The final
unqueried records are explicitly pending, not replaced by zero durations.
"""
import functools
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import torch


def batch_identity(batch):
    """Only CPU metadata/shape; no CUDA tensor contents or synchronization."""
    if batch is None:
        return None
    payload = {"requests": [(r.rid, len(r.output_ids)) for r in batch.reqs],
               "phase": str(batch.forward_mode),
               "M": int(batch.input_ids.numel()) if batch.input_ids is not None else 0,
               "prefix_lens": getattr(batch, "prefix_lens", None),
               "extend_lens": getattr(batch, "extend_lens", None),
               "seq_lens_sum": getattr(batch, "seq_lens_sum", None)}
    # Native ScheduleBatch keeps these as CPU Python lists, unlike seq_lens.
    for key in ("prefix_lens", "extend_lens"):
        assert payload[key] is None or isinstance(payload[key], (list, tuple))
    serialized = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def install(scheduler):
    root = Path(os.environ["SUCCESSOR_FASTPP_TRACE"])
    root.mkdir(parents=True, exist_ok=True)
    rank = int(scheduler.pp_rank)
    sink = (root / f"rank{rank}_pid{os.getpid()}.jsonl").open("a", buffering=1)
    pending = deque()
    current = None
    sequence = 0
    sample_every = int(os.environ.get("SUCCESSOR_TRACE_EVERY", "1"))
    sample_mode = os.environ.get("SUCCESSOR_SAMPLE_MODE", "local_sequence")
    assert sample_mode in {"local_sequence", "request_key", "all_stage_sparse_layers"}
    cap = int(os.environ.get("SUCCESSOR_TRACE_CAP", "4000"))
    runner = scheduler.tp_worker.model_runner
    sink.write(json.dumps({"kind": "instrumentation", "pid": os.getpid(),
                           "pp_rank": rank, "tp_rank": scheduler.tp_rank,
                           "device": str(scheduler.device),
                           "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "sample_every": sample_every, "cap": cap,
                           "sample_mode": sample_mode,
                           "cuda_timing": "same_device_events_no_cross_rank_subtraction",
                           "model": type(runner.model).__name__,
                           "runtime_transport_env": {key: os.environ.get(key) for key in
                               ("NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE")},
                           "moe_modules": [{"name": name, "class": type(module).__name__,
                                            "num_experts": getattr(module, "num_experts", None),
                                            "top_k": getattr(module, "top_k", None),
                                            "tp_size": getattr(module, "tp_size", None),
                                            "intermediate_per_partition": getattr(module, "intermediate_size_per_partition", None)}
                                           for name, module in runner.model.named_modules()
                                           if type(module).__name__ == "FusedMoE"]}) + "\n")

    def flush_ready():
        while pending and pending[0][2].query():
            record, start, end = pending.popleft()
            record["cuda_ms"] = start.elapsed_time(end)
            sink.write(json.dumps(record) + "\n")

    def timed(function, stage):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if current is None:
                return function(*args, **kwargs)
            if stage.startswith("layer:") and not current.get("layer_sampled", True):
                return function(*args, **kwargs)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            host_start = time.perf_counter()
            start.record()
            result = function(*args, **kwargs)
            end.record()
            pending.append(({**current, "stage": stage,
                             "host_start_s": host_start,
                             "host_end_s": time.perf_counter()}, start, end))
            return result
        return wrapped

    runner.forward = timed(runner.forward, "stage_compute")
    for name, module in runner.model.named_modules():
        if type(module).__name__.endswith("DecoderLayer"):
            module.forward = timed(module.forward, "layer:" + name)

    # Record the ORIGINAL predictor's values, before learning from an outcome.
    # These are host scheduler-loop predictions, not expert CUDA predictions.
    alp = getattr(scheduler, "alp_scheduler", None)
    if alp is not None:
        predict = alp.predict_exec_time_for_chunk
        learn = alp.update_runtime_overhead

        def predict_record(chunk_size, *args, **kwargs):
            result = predict(chunk_size, *args, **kwargs)
            if sequence <= cap:
                sink.write(json.dumps({"kind": "alp_prediction", "pp_rank": rank,
                                       "after_local_invocation_id": sequence,
                                       "host_s": time.perf_counter(), "chunk": chunk_size,
                                       "features": kwargs, "predicted_host_s": result}) + "\n")
            return result

        def learn_record(observed_exec_time, runtime_chunk_size=None,
                         decode_ctx_sum=0.0, prefill_square_sum=0.0,
                         prefill_prefix_prod_sum=0.0):
            features = {"decode_ctx_sum": decode_ctx_sum,
                        "prefill_square_sum": prefill_square_sum,
                        "prefill_prefix_prod_sum": prefill_prefix_prod_sum}
            predicted = None
            # Official startup creates zero-valued entries before profiling.
            # Its learn() intentionally returns without predicting in this state.
            if float(alp.prefill_exec_time_by_chunk.get(runtime_chunk_size, 0.0)) > 0.0:
                predicted = predict(runtime_chunk_size, **features)
            if sequence <= cap:
                sink.write(json.dumps({"kind": "alp_learning_sample", "pp_rank": rank,
                                       "after_local_invocation_id": sequence,
                                       "host_s": time.perf_counter(),
                                       "chunk": runtime_chunk_size, "features": features,
                                       "observed_host_s": observed_exec_time,
                                       "predicted_before_update_host_s": predicted}) + "\n")
            return learn(observed_exec_time, runtime_chunk_size, **features)

        alp.predict_exec_time_for_chunk = predict_record
        alp.update_runtime_overhead = learn_record

    original = scheduler.run_batch

    @functools.wraps(original)
    def run_batch(batch, pipe_batch):
        nonlocal current, sequence
        flush_ready()
        sequence += 1
        identity = batch_identity(batch) if sample_mode != "local_sequence" else None
        sample_index = int(identity[:16], 16) if identity else sequence
        selected = sample_index % sample_every == 0 and sequence <= cap
        if sample_mode == "request_key" and identity is None:
            selected = False  # Pipeline-only broadcast is not model execution.
        if sample_mode == "all_stage_sparse_layers":
            selected = identity is not None and sequence <= cap
        host_start = time.perf_counter()
        if selected:
            request_ids = [r.rid for r in batch.reqs] if batch is not None else []
            current = {"kind": "stage", "pp_rank": rank,
                       "tp_rank": scheduler.tp_rank,
                       "local_invocation_id": sequence,
                       "request_shape_key": identity,
                       "layer_sampled": sample_index % sample_every == 0,
                       "scheduler_forward_ct": scheduler.forward_ct,
                       "pipeline_slot": scheduler.cur_pp.get(),
                       "request_ids": request_ids,
                       "request_decode_positions": [len(r.output_ids) for r in batch.reqs]
                       if batch is not None else [],
                       "pipe_request_ids": [r.rid for r in pipe_batch.reqs]
                       if pipe_batch is not None else [],
                       "phase": str(batch.forward_mode) if batch is not None else "PIPE_ONLY",
                       "M": int(batch.input_ids.numel())
                       if batch is not None and batch.input_ids is not None else 0,
                       "chunk": scheduler.chunked_prefill_size}
        try:
            return original(batch, pipe_batch)
        finally:
            if current is not None:
                sink.write(json.dumps({**current, "kind": "run_batch_host",
                                       "host_start_s": host_start,
                                       "host_end_s": time.perf_counter(),
                                       "pending_events": len(pending)}) + "\n")
            current = None
            flush_ready()
    scheduler.run_batch = run_batch
