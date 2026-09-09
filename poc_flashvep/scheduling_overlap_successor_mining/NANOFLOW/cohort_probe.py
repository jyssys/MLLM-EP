"""Real-content fixed-cohort request probe using the official native workers.

This deliberately does not invent a continuous-batching scheduler. Each request
has an actual submission, first token and completion; no layer-to-E2E projection.
Model initialization and two exact-shape prefill warmups precede measurement.
Plan setup/graph capture on the measured request path remain charged.
"""
import json
from contextlib import nullcontext
import os
import time


def run(entry, path, output, forced_output=None):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    count = len(rows)
    assert count and len({len(r["input_ids"]) for r in rows}) == 1
    lengths = {r["max_new_tokens"] for r in rows}
    assert len(lengths) == 1
    output_count = next(iter(lengths))
    assert output_count >= 1
    forced = None
    if forced_output:
        forced = {r["request_id"]: r for r in json.loads(forced_output.read_text())["requests"]}
        for row in rows:
            assert row["input_ids"] == forced[row["request_id"]]["input_ids"]
            assert len(forced[row["request_id"]]["output_ids"]) == output_count
    # Official pool: 2048 pages × 16 tokens. Keep both warmup cohorts resident.
    required_pages = 2 * count * ((len(rows[0]["input_ids"]) + 15) // 16)
    required_pages += count * ((len(rows[0]["input_ids"]) + output_count + 15) // 16)
    assert required_pages < 2048, required_pages
    entry.command.value = b"Execute"
    entry.decode_bts.value = 0
    for warmup in range(2):
        inputs = [(warmup * count + i, row["input_ids"]) for i, row in enumerate(rows)]
        for queue in entry.request_queues:
            queue.put((inputs, None))
        entry.step_barrier(entry.barrier)
        assert len(entry.result_queue.get(timeout=30)) == count

    internal_to_row = {2 * count + i: row for i, row in enumerate(rows)}
    records = {index: {"request_id": row["request_id"], "input_ids": row["input_ids"],
                       "source_ids": row.get("source_ids", []),
                       "token_times_s": [], "output_ids": [], "arrival_s": 0.0}
               for index, row in internal_to_row.items()}
    inputs = [(index, row["input_ids"]) for index, row in internal_to_row.items()]
    start_unix = time.time()
    origin = time.perf_counter()
    for step in range(output_count):
        if os.environ.get("SUCCESSOR_NANO_NVTX") == "1":
            import nvtx
            marker = nvtx.annotate(f"COHORT_TARGET_STEP_{step:03d}")
        else:
            marker = nullcontext()
        with marker:
            for queue in entry.request_queues:
                queue.put((inputs, None))
            entry.step_barrier(entry.barrier)
            results = entry.result_queue.get(timeout=30)
        ready = time.perf_counter() - origin
        assert {i for i, _ in results} == set(records)
        for index, tokens in results:
            assert len(tokens) == 1
            records[index]["output_ids"].extend(tokens)
            records[index]["token_times_s"].append(ready)
        inputs = ([(index, [forced[row["request_id"]]["output_ids"][step]])
                   for index, row in internal_to_row.items()] if forced is not None else results)
        entry.decode_bts.value = count
    end_unix = time.time()
    for record in records.values():
        times = record["token_times_s"]
        record.update(ttft_s=times[0], e2e_s=times[-1],
                      tpot_s=(times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
                      itl_s=[b-a for a, b in zip(times, times[1:])])
        record["output_text"] = entry.arts.tokenizer.tokenizer.decode(record["output_ids"])
    output.write_text(json.dumps({
        "requests": list(records.values()), "start_unix": start_unix, "end_unix": end_unix,
        "tokens_s": count * output_count / (end_unix-start_unix),
        "request_contract": "simultaneous real-content cohort, not continuous batching",
        "timing": "parent request submission through actual native worker output receipt",
        "warmup": ("2 exact-shape prefill cohorts; prefill plan initialized during warmup"
                   if output_count == 1 else
                   "2 exact-shape prefill cohorts; first decode plan setup/capture charged to target"),
        "required_kv_pages": required_pages,
        "native_worker_and_numerical_ops_unchanged": True,
        "teacher_forced_from": str(forced_output) if forced_output else None,
        "performance_eligible": forced_output is None,
    }, indent=2))
    entry.terminate_workers(entry.processes, entry.barrier)
