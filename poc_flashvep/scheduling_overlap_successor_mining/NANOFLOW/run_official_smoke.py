"""Official dev-h100 correctness entry with only model/cache path adaptation.

Default execution loop and all workers use the official implementation.
Optional cohort mode changes the parent input loop only and is explicitly a
fixed-cohort transfer diagnostic, not an official online scheduler reproduction.
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from nanoflow.entry import test_multi_gpu as entry
from nanoflow.entry.common import CliArgs
from nanoflow.entry import common
from transformers import AutoTokenizer


class RecordingTokenizer:
    def __init__(self, tokenizer, output):
        self.tokenizer = tokenizer
        self.output = output

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def batch_decode(self, token_ids, **kwargs):
        decoded = self.tokenizer.batch_decode(token_ids, **kwargs)
        self.output.write_text(json.dumps({
            "token_ids_including_prompt": token_ids,
            "decoded": decoded,
            "prompt": "Hi, who are you?",
            "prompt_ids": self.encode("Hi, who are you?"),
            "entry": "official test_multi_gpu.test_correctness",
            "semantics_patch": False,
            "compatibility_patch": "DistKVPool(0, num_layers, ...) layer-range API",
        }, indent=2))
        return decoded


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nano-parts", type=int, choices=[1, 2, 4])
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cohort-trace", type=Path)
    parser.add_argument("--forced-output", type=Path)
    parser.add_argument("--capture-logits", type=Path)
    parser.add_argument("--plan-phase", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--compute-sms", type=int, choices=[*range(8, 128, 8), 132], default=132)
    parser.add_argument("--collective-sms", type=int, choices=[*range(8, 128, 8), 132], default=132)
    options = parser.parse_args()
    os.environ["SUCCESSOR_NANO_PLAN_PHASE"] = options.plan_phase
    os.environ["SUCCESSOR_NANO_COMPUTE_SMS"] = str(options.compute_sms)
    os.environ["SUCCESSOR_NANO_COLLECTIVE_SMS"] = str(options.collective_sms)
    if options.plan_phase == "prefill":
        assert options.cohort_trace and not options.cuda_graph, "Initial fixed-prefill diagnostic is eager only"
        assert all(json.loads(line)["max_new_tokens"] == 1 for line in options.cohort_trace.read_text().splitlines())
    assert (options.cache / "PACKING_COMPLETE.json").is_file()
    assert not options.out.exists()
    worker_entry = entry.worker_entry
    if options.nano_parts is not None:
        from nano_plan_adapter import worker_with_plan
        os.environ["SUCCESSOR_NANO_PARTS"] = str(options.nano_parts)
        os.environ["SUCCESSOR_NANO_PLAN_OUT"] = str(options.out.with_suffix(".plan"))
        worker_entry = worker_with_plan
    if options.forced_output or options.capture_logits:
        assert options.cohort_trace is not None
    if options.capture_logits:
        from numerical_probe import worker_with_checks
        os.environ["SUCCESSOR_NANO_LOGITS_OUT"] = str(options.capture_logits.resolve())
        os.environ["SUCCESSOR_NANO_COHORT_COUNT"] = str(len(options.cohort_trace.read_text().splitlines()))
        worker_entry = worker_with_checks
    mp.set_start_method("spawn")
    torch.set_num_threads(8)
    entry.args = CliArgs(model="Qwen1.5-MoE-A2.7B-EP", expert_parallel_size=4)
    entry.world_size = entry.world_info()
    assert entry.world_size == 4
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    # The pinned snapshot exists locally without an unpinned refs/main entry.
    # Adapt only the official factory's tokenizer path, never model semantics.
    with patch.object(common, "AutoTokenizer", SimpleNamespace(
            from_pretrained=lambda *a, **k: tokenizer)):
        entry.arts = entry.setup_model_and_configs(entry.args)
    for cfg in entry.arts.cfgs:
        cfg.cached_weight_dir = str(options.cache.resolve())
    entry.pipeline_list = entry.create_pipelines(entry.arts.cfgs, entry.arts.Pipeline)
    (entry.command, entry.decode_bts, entry.next_decode_bts,
     entry.auto_search_enabled, entry.nano_split_enabled,
     entry.plan_cuda_graph, entry.cuda_graph_enabled,
     entry.plan_double_buffer, entry.double_buffer_enabled,
     entry.barrier) = entry.create_shared_variables(4)
    entry.request_queues = [mp.Queue(maxsize=1000) for _ in range(4)]
    entry.result_queue = mp.Queue(maxsize=1000)
    start = time.time()
    entry.processes = entry.start_workers(
        0.0, 4, None, entry.request_queues, entry.decode_bts,
        entry.next_decode_bts, entry.result_queue, entry.barrier,
        entry.pipeline_list, entry.auto_search_enabled,
        entry.arts.auto_search_path, entry.nano_split_enabled,
        entry.plan_cuda_graph, entry.cuda_graph_enabled,
        entry.plan_double_buffer, entry.double_buffer_enabled,
        entry.command, worker_entry)
    entry.arts.tokenizer = RecordingTokenizer(entry.arts.tokenizer, options.out)
    step_timings = []
    target_decode_count = (len(options.cohort_trace.read_text().splitlines())
                           if options.cohort_trace else 4)
    original_step_barrier = entry.step_barrier

    def timed_step_barrier(barrier):
        # Capture only after the official smoke reaches its stable decode cohort.
        if options.cuda_graph and entry.decode_bts.value == target_decode_count:
            first_fixed_decode = not any(x["decode_requests"] == target_decode_count for x in step_timings)
            entry.plan_cuda_graph.value = first_fixed_decode
            entry.cuda_graph_enabled.value = not first_fixed_decode
        step_start = time.time()
        original_step_barrier(barrier)
        step_timings.append({"step": len(step_timings),
                             "command": entry.command.value.decode(),
                             "decode_requests": entry.decode_bts.value,
                             "start_unix": step_start, "end_unix": time.time(),
                             "includes_cpu_and_gpu": True,
                             "plan_cuda_graph": entry.plan_cuda_graph.value,
                             "cuda_graph_enabled": entry.cuda_graph_enabled.value,
                             "first_step_may_include_init_jit": len(step_timings) == 0})
    entry.step_barrier = timed_step_barrier
    finished = threading.Event()

    def watch_workers():
        while not finished.wait(1):
            if any(p.exitcode not in (None, 0) for p in entry.processes):
                entry.barrier.abort()
                return

    threading.Thread(target=watch_workers, daemon=True).start()
    try:
        if options.cohort_trace:
            from cohort_probe import run
            run(entry, options.cohort_trace, options.out, options.forced_output)
        else:
            entry.test_correctness()
    finally:
        finished.set()
        for process in entry.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        options.out.with_suffix(".run_meta.json").write_text(json.dumps({
            "start_unix": start, "end_unix": time.time(),
            "physical_gpus": [4, 5, 6, 7],
            "worker_exit_codes": [p.exitcode for p in entry.processes],
            "result_exists": options.out.exists(),
            "dtype": "float16_official_native_path",
            "purpose": "official_baseline_correctness_not_performance",
            "manual_plan_parts": options.nano_parts,
            "fixed_plan_phase": options.plan_phase,
            "fixed_decode_cuda_graph": options.cuda_graph,
            "cohort_trace": str(options.cohort_trace) if options.cohort_trace else None,
            "forced_output": str(options.forced_output) if options.forced_output else None,
            "captured_logits": str(options.capture_logits) if options.capture_logits else None,
            "performance_eligible": options.forced_output is None and options.capture_logits is None,
            "step_timings": step_timings,
        }, indent=2))


if __name__ == "__main__":
    main()
