"""Natural online decode interference on the unchanged RLT execution path."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from .common import ROOT, MODEL, guard, configuration, write

OLD = Path(
    "/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/dataset_manifest.json"
)


def freeze():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=True
    )
    source = json.loads(OLD.read_text())["rows"]
    cases = []
    for trial in range(16):
        questions = [
            source[(trial * 31 + i * 13) % len(source)]["question"] for i in range(100)
        ]
        text = "Mathematical reference problems:\n" + "\n\n".join(
            f"Problem {i + 1}: {q}" for i, q in enumerate(questions)
        )
        ids = tokenizer.encode(text)
        for length in (2048, 8192):
            expanded = (ids * ((length + len(ids) - 1) // len(ids)))[: length - 32]
            ending = tokenizer.encode(
                "\nExplain the reasoning for these tasks and provide a clear mathematical summary.\nAnswer:"
            )
            prompt = expanded + ending
            # Natural text tokens, with a frozen length bin. Never random token IDs.
            cases.append(
                dict(
                    id=f"long_{trial}_{length}",
                    trial=trial,
                    prompt_bin=length,
                    input_ids=prompt,
                    text=tokenizer.decode(prompt),
                    construction="Concatenated public mathematical problem text; bounded context excerpts.",
                )
            )
        for i in range(7):
            row = source[(trial * 17 + i) % len(source)]
            prompt = tokenizer.encode(row["prompt"])
            cases.append(
                dict(
                    id=f"bg_{trial}_{i}",
                    trial=trial,
                    background_index=i,
                    input_ids=prompt,
                    text=row["prompt"],
                    source_id=row["id"],
                )
            )
    manifest = dict(
        model=str(MODEL),
        seed=20261022,
        cases=cases,
        concurrency=[4, 8],
        chunks=[128, 256, 512, 1024],
        groups=[1, 2, 4],
        max_new_tokens=128,
        temperature=0,
        EOS=True,
        arrival="One long request at fixed 20ms after a primed decode cohort origin.",
        calibration_trials=[0, 1],
        heldout_trials=list(range(2, 16)),
        gates=dict(
            ITL_p95_improvement=0.05,
            goodput_improvement=0.10,
            TTFT_allowed_regression=0.05,
        ),
        numerical=dict(
            expect_exact_same_C=True,
            hidden_rel_l2_max=0.03,
            logits_rel_l2_max=0.03,
            kv_rel_l2_max=0.03,
        ),
        SLO=dict(TTFT_s=5.0, request_max_ITL_s=0.100),
        clock="In-process client monotonic arrival-to-output clock; no HTTP or network claims.",
    )
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    target = ROOT / "workload_manifest.json"
    assert not target.exists()
    write(target, manifest)
    return manifest


class Observe:
    def __init__(self, engine, trace, record=True):
        self.engine = engine
        self.trace = trace
        self.records = []
        self.pending = []
        self.current = None
        self.original_submit = engine.executor.submit
        self.original_forward = engine.model_runner.runtime.forward
        self.anchor = torch.cuda.Event(enable_timing=True) if trace else None
        self.clock = None
        if self.anchor:
            engine.model_runner.synchronize()
            before = time.perf_counter()
            self.anchor.record()
            self.anchor.synchronize()
            after = time.perf_counter()
            self.clock = dict(
                host_midpoint_s=(before + after) / 2, uncertainty_s=(after - before) / 2
            )
        if record or trace:
            engine.executor.submit = self.submit
        if trace:
            engine.model_runner.runtime.forward = self.forward

    def submit(self, batch):
        self.current = batch
        start = time.perf_counter()
        if self.trace:
            from vllm_rlt.worker.executor import build_execution_batch

            descriptor = build_execution_batch(batch, 4)
            stream = (
                self.engine.model_runner.core_stream
                if descriptor.loop_count
                else self.engine.model_runner.boundary_stream
            )
            begin_event, end_event = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin_event.record(stream)
            torch.cuda.nvtx.range_push(
                "CHUNK_LOOP|"
                + ",".join(batch.phase_for(r).value for r in batch.requests)
            )
        ticket = self.original_submit(batch)
        if self.trace:
            end_event.record(stream)
            torch.cuda.nvtx.range_pop()
            self.pending.append(
                (
                    begin_event,
                    end_event,
                    dict(
                        quantum=len(self.records),
                        operation="quantum_with_staging_and_stream_waits",
                        ids=[r.request.request_id for r in batch.requests],
                    ),
                )
            )
        self.records.append(
            dict(
                host_begin=start,
                host_submit_end=time.perf_counter(),
                spans=[
                    dict(
                        id=r.request.request_id,
                        phase=batch.phase_for(r).value,
                        start=r.token_start,
                        tokens=r.token_count,
                        depth=r.depth,
                        group=(
                            4
                            if batch.phase_for(r).value == "prefill"
                            else getattr(r, "loop_count", 1)
                            if batch.phase_for(r).value
                            in ("recurrent", "prefilling_recurrent")
                            else 0
                        ),
                    )
                    for r in batch.requests
                ],
            )
        )
        return ticket

    def forward(self, batch):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        result = self.original_forward(batch)
        end.record()
        self.pending.append(
            (
                start,
                end,
                dict(
                    submission=len(self.records),
                    operation=batch.operation.value,
                    recurrent_calls=len(batch.metadata)
                    if hasattr(batch.metadata, "__len__")
                    else None,
                    ids=[r.request.request_id for r in self.current.requests],
                ),
            )
        )
        return result

    def finish(self):
        self.engine.model_runner.synchronize()
        for start, end, row in self.pending:
            row["gpu_compute_ms"] = start.elapsed_time(end)
            row["gpu_start_ms"] = self.anchor.elapsed_time(start)
            row["gpu_end_ms"] = self.anchor.elapsed_time(end)
        return [row for _, _, row in self.pending]


@torch.inference_mode()
def run(
    model,
    mode,
    chunk,
    length,
    concurrency,
    trial,
    trace,
    graphs,
    record=True,
    engine=None,
):
    from vllm_rlt import SamplingParams
    from vllm_rlt.engine.stage_engine import StageEngine

    manifest = json.loads((ROOT / "workload_manifest.json").read_text())
    cases = manifest["cases"]
    params = SamplingParams(max_tokens=128, max_loops=4, exit_threshold=1.0)
    reused = engine is not None
    if engine is None:
        engine = StageEngine(
            model, **configuration(mode, chunk, concurrency, graphs=graphs)
        )
    observer = Observe(engine, trace, record)
    token_times = {}
    tokens = {}
    depths = {}
    finished = {}
    arrivals = {}
    warmup_origin = time.perf_counter()
    background = [r for r in cases if r["trial"] == trial and "background_index" in r][
        : concurrency - 1
    ]
    for row in background:
        rid = row["id"]
        engine.add_request(rid, row["input_ids"], params)
        token_times[rid] = []
        tokens[rid] = []
        arrivals[rid] = warmup_origin
    primed = False
    long_added = False
    origin = None
    long_arrival = None
    live_at_admission = 0
    while engine.has_unfinished_requests() or not long_added:
        now = time.perf_counter()
        if not primed and all(len(token_times[r["id"]]) >= 1 for r in background):
            primed = True
            origin = now
            long_arrival = origin + 0.020
        if primed and not long_added and now >= long_arrival:
            row = next(
                r
                for r in cases
                if r["trial"] == trial and r.get("prompt_bin") == length
            )
            rid = row["id"]
            live_at_admission = sum(
                r["id"] in engine.scheduler.requests for r in background
            )
            engine.add_request(rid, row["input_ids"], params)
            arrivals[rid] = long_arrival
            token_times[rid] = []
            tokens[rid] = []
            long_added = True
        if not engine.has_unfinished_requests():
            continue
        for output in engine.step():
            now = time.perf_counter()
            rid = output.request_id
            previous = len(tokens[rid])
            new = list(output.token_ids)
            if len(new) > previous:
                assert len(new) == previous + 1, (
                    "No invented per-token timestamps for chunks."
                )
                token_times[rid].append(now)
                tokens[rid] = new
                depths[rid] = list(output.exit_depths)
            if output.finished:
                finished[rid] = dict(reason=output.finish_reason, complete=now)
    gpu = observer.finish()
    end = time.perf_counter()
    rows = []
    for rid, times in token_times.items():
        record = dict(
            id=rid,
            arrival=arrivals[rid],
            token_times=times,
            output_ids=tokens[rid],
            depths=depths[rid],
            prompt_tokens=len(next(r for r in cases if r["id"] == rid)["input_ids"]),
            finish_reason=finished[rid]["reason"],
            E2E_s=finished[rid]["complete"] - arrivals[rid],
            TTFT_s=times[0] - arrivals[rid],
            TPOT_s=(times[-1] - times[0]) / (len(times) - 1)
            if len(times) > 1
            else None,
            decode_ITLs_s=[b - a for a, b in zip(times, times[1:])],
            measured_ITLs_s=[b - a for a, b in zip(times, times[1:]) if origin <= b],
            background=rid.startswith("bg_"),
        )
        rows.append(record)
    graphs_obj = engine.model_runner.graphs
    result = dict(
        mode=mode,
        chunk_tokens=chunk,
        loop_group=4
        if mode in ("atomic", "staged4")
        else 2
        if mode == "grouped"
        else 1,
        backend=engine.cache_manager.backend,
        backend_info=engine.cache_manager.attention_info,
        prompt_bin=length,
        concurrency=concurrency,
        trial=trial,
        heavy=trace,
        graphs=graphs,
        origin=origin,
        end=end,
        measured_cohort_s=end - origin,
        priming_s=origin - warmup_origin,
        client_cohort_makespan_s=end - warmup_origin,
        record_quanta=record,
        engine_reused=reused,
        trace_clock=observer.clock,
        live_decode_at_long_admission=live_at_admission,
        requests=rows,
        quantums=observer.records,
        gpu_quantums=gpu,
        graph_entries=len(getattr(graphs_obj, "entries", {})) if graphs_obj else 0,
        graph_stats={
            k: getattr(graphs_obj, k, 0) for k in ("captures", "replays", "fallbacks")
        }
        if graphs_obj
        else {},
        memory=dict(
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            peak=torch.cuda.max_memory_allocated(),
            KV_bytes=engine.cache_manager.key_cache.numel()
            * engine.cache_manager.key_cache.element_size()
            * 2,
            activation_bytes=engine.model_runner.prefill_states.numel() * 2
            if engine.model_runner.prefill_states is not None
            else 0,
        ),
    )
    engine.executor.submit = observer.original_submit
    engine.model_runner.runtime.forward = observer.original_forward
    del observer
    if not reused:
        engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--label")
    p.add_argument("--modes", default="atomic,split,mixed")
    p.add_argument("--chunks", default="128,256,512,1024")
    p.add_argument("--lengths", default="2048")
    p.add_argument("--concurrencies", default="4")
    p.add_argument("--trials", default="0")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--cells", help="Frozen mode:chunk pairs for confirmation.")
    p.add_argument("--shapes", help="Frozen prompt:concurrency pairs.")
    p.add_argument("--no-record-quanta", action="store_true")
    p.add_argument("--warmup-trial", type=int)
    p.add_argument(
        "--reuse-engine",
        action="store_true",
        help="Keep each configuration resident across trial cohorts.",
    )
    a = p.parse_args()
    if a.warmup_trial is not None and not a.reuse_engine:
        p.error("--warmup-trial requires --reuse-engine")
    if a.freeze:
        freeze()
        return
    guard()
    torch.set_num_threads(1)
    from vllm_rlt.models import OuroForCausalLM

    model = OuroForCausalLM.from_pretrained(
        MODEL, device="cuda:0", dtype=torch.bfloat16
    )
    target = ROOT / "raw" / a.label
    target.mkdir(parents=True, exist_ok=False)
    cells = (
        [(m, int(c)) for m, c in (s.split(":") for s in a.cells.split(","))]
        if a.cells
        else [(m, int(c)) for c in a.chunks.split(",") for m in a.modes.split(",")]
    )
    shapes = (
        [(int(p), int(n)) for p, n in (s.split(":") for s in a.shapes.split(","))]
        if a.shapes
        else [
            (int(p), int(n))
            for p in a.lengths.split(",")
            for n in a.concurrencies.split(",")
        ]
    )
    trials = list(map(int, a.trials.split(",")))
    jobs = (
        [(t, p, n, m, c) for p, n in shapes for m, c in cells for t in trials]
        if a.reuse_engine
        else [(t, p, n, m, c) for t in trials for p, n in shapes for m, c in cells]
    )
    resident = None
    resident_key = None
    for trial, length, concurrency, mode, chunk in jobs:
        if a.reuse_engine and resident_key != (mode, chunk, concurrency, length):
            from vllm_rlt.engine.stage_engine import StageEngine

            if resident is not None:
                resident.close()
                del resident
                gc.collect()
                torch.cuda.empty_cache()
            resident = StageEngine(
                model, **configuration(mode, chunk, concurrency, graphs=not a.no_graphs)
            )
            resident_key = (mode, chunk, concurrency, length)
            if a.warmup_trial is not None:
                assert a.warmup_trial not in trials
                print("WARMUP", mode, chunk, length, concurrency, flush=True)
                warm = run(
                    model,
                    mode,
                    chunk,
                    length,
                    concurrency,
                    a.warmup_trial,
                    False,
                    not a.no_graphs,
                    False,
                    resident,
                )
                warm["warmup"] = True
                write(
                    target / f"warmup_{mode}_C{chunk}_P{length}_N{concurrency}.json",
                    warm,
                )
        label = f"{mode}_C{chunk}_P{length}_N{concurrency}_T{trial}"
        print("RUN", label, flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            result = run(
                model,
                mode,
                chunk,
                length,
                concurrency,
                trial,
                a.trace,
                not a.no_graphs,
                not a.no_record_quanta,
                resident,
            )
            write(target / (label + ".json"), result)
            itls = [
                v
                for r in result["requests"]
                if r["background"]
                for v in r["measured_ITLs_s"]
            ]
            print(
                "RESULT",
                label,
                "cohort",
                round(result["measured_cohort_s"], 3),
                "ITL95",
                round(float(np.percentile(itls, 95)), 4) if itls else None,
                "longTTFT",
                round(
                    next(
                        r["TTFT_s"] for r in result["requests"] if not r["background"]
                    ),
                    4,
                ),
                "elapsed",
                round(time.perf_counter() - started, 2),
                flush=True,
            )
        except Exception as error:
            import traceback

            write(
                target / (label + "_FAIL.json"),
                dict(error=repr(error), traceback=traceback.format_exc()),
            )
            raise
    if resident is not None:
        resident.close()


if __name__ == "__main__":
    main()
