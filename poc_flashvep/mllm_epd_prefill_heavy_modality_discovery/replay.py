"""Replay observed full E-ready spacing through actual text P/D serving."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt


def freeze_ready_trace(proxy, tag, out):
    ready = []
    with proxy.open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row["runtime_request_id"].startswith(tag + "_")
                    and row.get("status") == "MEASURED_ONLINE" and "E_complete" in row):
                ready.append((row["E_complete"], row["runtime_request_id"]))
    ready.sort()
    if len(ready) < 2 or len(set(req for _, req in ready)) != len(ready):
        raise ValueError("Select a completed EPD cohort with distinct E-ready requests")
    start = ready[0][0]
    offsets = [moment - start for moment, _ in ready]
    out.write_text(json.dumps({"kind": "observed_e_ready", "source": str(proxy),
                               "source_SHA256": rt.sha(proxy), "tag": tag,
                               "request_ids": [req for _, req in ready], "offsets": offsets,
                               "qps": (len(offsets) - 1) / offsets[-1]}, indent=2) + "\n")


async def execute(client, rows, url, tag, trace, output, metadata):
    from ..mllm_epd_prefill_heavy_token_split.client_evidence import DeferredEvidence

    offsets = trace["offsets"]
    rt.require(len(rows) == len(offsets), "Replay and roster lengths differ")
    rt.require(offsets and offsets[0] == 0 and offsets == sorted(offsets), "Invalid open-loop offsets")
    observed = trace.get("kind", "observed_e_ready") == "observed_e_ready"
    rt.require(not observed or all(not row["images"] for row in rows), "E-ready replay requires text inputs")
    description = ("Observed full E readiness from a separate multimodal native run." if observed
                   else "Explicit controlled open-loop burst offsets.")
    root = client.ROOT
    schedule = [{"index": i, "request_id": row["request_id"], "offset_seconds": offset, "replica": 0}
                for i, (row, offset) in enumerate(zip(rows, offsets, strict=True))]
    timetable = root/"configs"/f"{tag}_arrival.json"
    rt.write(timetable, {"QPS": trace["qps"], "trace":schedule, "source": trace,
                         "definition": description})
    rt.write(root/"configs"/f"{tag}.json", {
        "urls":[url], "QPS":trace["qps"], "count":len(rows), "output_count":output,
        "output_mode":"Natural EOS" if output is None else "Fixed output", "prefix_cache":False,
        "arrival_SHA256":rt.sha(timetable), "transport":"HTTP SSE", "workload":metadata})
    sink = DeferredEvidence(root/"raw"/f"{tag}_requests.jsonl",
                            root/"raw"/f"{tag}_completion_journal.jsonl", None)
    start = time.monotonic() + 1
    wall_start = time.time()
    timeout = aiohttp.ClientTimeout(total=1800)
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0),
                                         trust_env=False) as session:
            results = await asyncio.gather(*(
                client.one(session, url, row, client.payload(row, f"{tag}_{i:04d}", output),
                           start + offset, tag, i, evidence_sink=sink)
                for i, (row, offset) in enumerate(zip(rows, offsets, strict=True))))
    finally:
        sink.flush()
    counts = sum(r["status"] == "MEASURED_ONLINE" for r in results)
    rt.write(root/"state"/f"{tag}_complete.json", {
        "run":tag, "wall_start":wall_start, "wall_end":time.time(), "monotonic_start":start,
        "planned_count":len(rows), "actual_count":len(results), "not_run_count":0, "completed":counts,
        "duration_seconds":max(r["finish"] for r in results)-start, "serial":False, "passed":counts==len(rows),
        "raw_SHA256":rt.sha(root/"raw"/f"{tag}_requests.jsonl")})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze_ready_trace(args.proxy, args.tag, args.out)


if __name__ == "__main__":
    main()
