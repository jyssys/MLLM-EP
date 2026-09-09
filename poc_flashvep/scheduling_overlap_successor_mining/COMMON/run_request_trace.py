"""Replay explicit arrivals to official native streaming endpoints.

Client-observed request latency is primary here. Stream coalescing is exposed,
not silently treated as an internal token-ready timestamp. No CUDA use.
"""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp
import numpy as np


async def run_one(session, row, origin, args):
    await asyncio.sleep(max(0, origin + row["arrival_s"] - time.perf_counter()))
    sent = time.perf_counter()
    params = {"temperature": 0.0, "max_new_tokens": row["max_new_tokens"],
              "ignore_eos": row.get("ignore_eos", False)}
    if args.backend == "fastpp":
        # Official GenerateReqInput.rid propagates into ScheduleBatch.reqs.
        payload = {"text": row["prompt"], "sampling_params": params, "stream": True,
                   "rid": args.run_id + ":" + row["request_id"]}
    else:
        payload = {"prompt": row["prompt"], "temperature": 0.0,
                   "max_tokens": row["max_new_tokens"],
                   "ignore_eos": row.get("ignore_eos", False), "stream": True}
    record = {k: v for k, v in row.items() if k != "prompt"}
    record.update(run_id=args.run_id, backend=args.backend,
                  sent_s=sent-origin, arrival_lateness_s=sent-origin-row["arrival_s"],
                  stream_events=[], status="RUNNING")
    if args.backend == "fastpp":
        record["server_request_id"] = payload["rid"]
    previous_count = 0
    final_text = ""
    final_ids = None
    pending = b""
    try:
        async with session.post(args.url + "/generate", json=payload,
                                headers={"X-Successor-Request-ID": args.run_id + ":" + row["request_id"]}) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_any():
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line = line.strip()
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    if not line or line == b"[DONE]":
                        continue
                    item = json.loads(line)
                    now = time.perf_counter()
                    if args.backend == "fastpp":
                        meta = item.get("meta_info", {})
                        count = int(meta.get("completion_tokens", 0))
                        final_text = item.get("text", final_text)
                        final_ids = item.get("output_ids", final_ids)
                        record["server_meta"] = meta
                    else:
                        # Official _AsyncLLMEngine.step emits only last_token,
                        # not cumulative text/IDs. Preserve every delta.
                        final_text += item.get("generated_text", "")
                        delta_ids = item.get("output_tokens", [])
                        if not isinstance(delta_ids, list):
                            raise TypeError("Layered stream output_tokens is not a list")
                        final_ids = (final_ids or []) + delta_ids
                        count = len(final_ids)
                    if count > previous_count:
                        record["stream_events"].append({"t_s": now-origin,
                                                        "count": count,
                                                        "delta": count-previous_count})
                        previous_count = count
            if pending.strip() not in (b"", b"data: [DONE]"):
                raise ValueError("Unparsed trailing stream bytes")
        record["status"] = "PASS" if previous_count else "EMPTY_OUTPUT"
    except Exception as exc:
        record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
    done = time.perf_counter()
    events = record["stream_events"]
    record.update(done_s=done-origin, e2e_s=done-sent,
                  output_tokens=previous_count, output_text=final_text,
                  output_ids=final_ids,
                  output_sha256=hashlib.sha256(final_text.encode()).hexdigest())
    if events:
        record["ttft_s"] = events[0]["t_s"] - (sent-origin)
        record["tpot_s"] = ((events[-1]["t_s"]-events[0]["t_s"])/(previous_count-1)
                            if previous_count > 1 else None)
        record["stream_coalesced"] = any(e["delta"] != 1 for e in events)
        record["itl_s"] = [b["t_s"]-a["t_s"] for a, b in zip(events, events[1:])]
    with args.out.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps({k: record.get(k) for k in
                      ("request_id", "status", "e2e_s", "ttft_s", "output_tokens")}), flush=True)
    return record


async def main(args):
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if args.out.exists():
        raise FileExistsError("Use a new output file per independent run")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout,
                                    connector=aiohttp.TCPConnector(limit=0)) as session:
        start_unix = time.time()
        origin = time.perf_counter()
        records = await asyncio.gather(*(run_one(session, row, origin, args) for row in rows))
        elapsed = time.perf_counter()-origin
    valid = [r for r in records if r["status"] == "PASS"]
    summary = {"run_id": args.run_id, "start_unix": start_unix,
               "elapsed_s": elapsed, "requests": len(rows), "valid": len(valid),
               "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
               "tokens_per_s": sum(r["output_tokens"] for r in valid)/elapsed,
               "requests_per_s": len(valid)/elapsed,
               "measurement": "client_observed_native_stream_not_internal_token_ready"}
    for key in ("e2e_s", "ttft_s", "tpot_s"):
        values = [r[key] for r in valid if r.get(key) is not None]
        summary[key] = ({"mean": float(np.mean(values)),
                         **{f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 99)}}
                        if values else None)
    summary["coalesced_requests"] = sum(r.get("stream_coalesced", False) for r in valid)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--backend", required=True, choices=["fastpp", "layered"])
    p.add_argument("--url", default="http://127.0.0.1:31800")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--limit", type=int)
    asyncio.run(main(p.parse_args()))
