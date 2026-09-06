"""Controlled OpenAI-compatible online workload generator.

Every block has a distinct causal question.  Requests can be pinned to a DP
rank with the supported X-data-parallel-rank header, enabling request-multiset
controls without modifying vLLM scheduling or model math.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import aiohttp


@dataclass(frozen=True)
class Req:
    prompt_words: int
    output_tokens: int
    dp_rank: int | None
    image: str | None = None
    delay_s: float = 0.0
    image2: str | None = None


def text_prompt(words: int, seed: int) -> str:
    topics = ["compiler", "database", "algebra", "history", "network", "biology"]
    body = " ".join(topics[(i + seed) % len(topics)] for i in range(words))
    return f"Explain the following material carefully: {body}"


@lru_cache(maxsize=8)
def _image_data(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


async def one(session: aiohttp.ClientSession, base: str, model: str, req: Req,
              request_id: str, seed: int) -> dict:
    if req.delay_s:
        await asyncio.sleep(req.delay_s)
    content: list[dict] = []
    if req.image:
        content.append({"type": "image_url", "image_url": {"url": _image_data(req.image)}})
    if req.image2:
        content.append({"type": "image_url", "image_url": {"url": _image_data(req.image2)}})
    content.append({"type": "text", "text": text_prompt(req.prompt_words, seed)})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": req.output_tokens,
        "stream": True,
    }
    headers = {"X-Request-Id": request_id}
    if req.dp_rank is not None:
        headers["X-data-parallel-rank"] = str(req.dp_rank)
    t0 = time.perf_counter_ns()
    first = None
    chunks = 0
    status = None
    error = None
    try:
        async with session.post(base + "/v1/chat/completions", json=payload, headers=headers) as resp:
            status = resp.status
            async for raw in resp.content:
                if raw.strip():
                    if first is None:
                        first = time.perf_counter_ns()
                    chunks += 1
            if resp.status != 200:
                error = (await resp.text())[:1000]
    except Exception as exc:
        error = repr(exc)
    t1 = time.perf_counter_ns()
    return {
        "request_id": request_id,
        "dp_rank": req.dp_rank,
        "prompt_words": req.prompt_words,
        "output_tokens_requested": req.output_tokens,
        "image_count": int(bool(req.image)) + int(bool(req.image2)),
        "has_image": bool(req.image or req.image2),
        "delay_s": req.delay_s,
        "status": status,
        "error": error,
        "chunks": chunks,
        "arrival_ns": t0,
        "first_chunk_ns": first,
        "finish_ns": t1,
        "ttft_ms": None if first is None else (first - t0) / 1e6,
        "e2e_ms": (t1 - t0) / 1e6,
    }


def blocks(images: list[str]) -> list[tuple[str, str, list[Req], list[Req]]]:
    """Return (hypothesis, variant, A/B request schedules)."""
    short, med, long = 48, 256, 960
    return [
        ("H03", "balanced_vs_dp_skew",
         [Req(med, 16, 0)] * 4 + [Req(med, 16, 1)] * 4,
         [Req(448, 16, 0)] * 4 + [Req(64, 16, 1)] * 4),
        ("H04", "phase_aligned_vs_misaligned",
         [Req(med, 1, 0)] * 2 + [Req(med, 63, 0)] * 2
         + [Req(med, 1, 1)] * 2 + [Req(med, 63, 1)] * 2,
         [Req(med, 63, 0)] * 4 + [Req(med, 1, 1)] * 4),
        ("H05", "both_active_vs_effectively_idle_dp",
         [Req(med, 24, 0)] * 3 + [Req(med, 1, 0)]
         + [Req(med, 24, 1)] * 3 + [Req(med, 1, 1)],
         [Req(med, 24, 0)] * 6 + [Req(med, 1, 1)] * 2),
        ("H06", "uniform_vs_output_churn",
         [Req(med, 16, 0), Req(med, 16, 0), Req(med, 16, 1), Req(med, 16, 1)],
         [Req(med, 1, 0), Req(med, 31, 0), Req(med, 1, 1), Req(med, 31, 1)]),
        ("H07", "within_vs_cross_dp_heterogeneity",
         [Req(long, 8, 0), Req(short, 8, 0), Req(long, 8, 1), Req(short, 8, 1)],
         [Req(long, 8, 0), Req(long, 8, 0), Req(short, 8, 1), Req(short, 8, 1)]),
        ("H08", "request_order",
         [Req(long, 8, 0), Req(short, 8, 0), Req(long, 8, 1), Req(short, 8, 1)],
         [Req(short, 8, 0), Req(long, 8, 0), Req(short, 8, 1), Req(long, 8, 1)]),
        ("H09", "chunk_residue",
         [Req(480, 4, 0)] * 4 + [Req(480, 4, 1)] * 4,
         [Req(520, 4, 0)] * 2 + [Req(440, 4, 0)] * 2
         + [Req(520, 4, 1)] * 2 + [Req(440, 4, 1)] * 2),
        ("H10", "few_long_vs_many_short",
         [Req(960, 1, 0)] * 2 + [Req(960, 1, 1)] * 2,
         [Req(240, 1, 0)] * 8 + [Req(240, 1, 1)] * 8),
        ("H11", "separated_vs_staggered_mixed_phase",
         [Req(long, 1, 0)] * 2 + [Req(long, 1, 1)] * 2
         + [Req(short, 64, 0, delay_s=2.0)] * 2
         + [Req(short, 64, 1, delay_s=2.0)] * 2,
         [Req(short, 64, 0)] * 2 + [Req(short, 64, 1)] * 2
         + [Req(long, 1, 0, delay_s=0.08)] * 2
         + [Req(long, 1, 1, delay_s=0.08)] * 2),
        ("H12", "steady_vs_bursty",
         [Req(med, 16, None, delay_s=i * 0.04) for i in range(8)],
         [Req(med, 16, None)] * 8),
        ("H14", "short_vs_long_attention_context",
         [Req(short, 96, None)] * 4,
         [Req(long, 96, None)] * 4),
        ("H15", "text_vs_vision_predecessor",
         [Req(med, 8, None)] * 4,
         [Req(40, 8, None, image=images[0])] * 4),
        ("H23", "one_image_vs_two_images",
         [Req(40, 8, None, image=images[0])] * 4,
         [Req(40, 8, None, image=images[0], image2=images[1])] * 4),
        ("H25", "dense_vs_paced_arrival",
         [Req(med, 16, None)] * 8,
         [Req(med, 16, None, delay_s=i * 0.02) for i in range(8)]),
        ("H28", "stable_set_vs_turnover",
         [Req(med, 32, None)] * 8,
         [Req(med, 1 if i % 2 == 0 else 63, None) for i in range(8)]),
        ("H36", "idle_dp_vs_short_participant",
         [Req(med, 24, 0)] * 6 + [Req(med, 1, 1)] * 2,
         [Req(med, 24, 0)] * 6),
        ("H37", "one_dp_output_churn",
         [Req(med, 16, 0)] * 4 + [Req(med, 16, 1)] * 4,
         [Req(med, 1, 0), Req(med, 31, 0)] * 2
         + [Req(med, 16, 1)] * 4),
        ("H38", "metadata_step_recurrence",
         [Req(med, 1, 0)] * 4 + [Req(med, 1, 1)] * 4,
         [Req(med, 16, 0)] * 4 + [Req(med, 16, 1)] * 4),
        ("H43", "pinned_vs_unpinned_dp",
         [Req(med, 8, 0)] * 4 + [Req(med, 8, 1)] * 4,
         [Req(med, 8, None)] * 8),
        # Source-derived controls.  These reuse an already audited equal-work
        # request multiset; the runtime restart/configuration is the causal
        # intervention, not a new workload shape.
        ("H31", "async_scheduler_control",
         [Req(med, 24, 0)] * 3 + [Req(med, 1, 0)]
         + [Req(med, 24, 1)] * 3 + [Req(med, 1, 1)],
         [Req(med, 24, 0)] * 6 + [Req(med, 1, 1)] * 2),
        ("H32", "dp_sync_backend_control",
         [Req(med, 16, 0)] * 4 + [Req(med, 16, 1)] * 4,
         [Req(448, 16, 0)] * 4 + [Req(64, 16, 1)] * 4),
        ("H33", "metadata_host_path_control",
         [Req(med, 16, 0), Req(med, 16, 0), Req(med, 16, 1), Req(med, 16, 1)],
         [Req(med, 1, 0), Req(med, 31, 0), Req(med, 1, 1), Req(med, 31, 1)]),
        ("H45", "common_regime_transition_repeat",
         [Req(480, 4, 0)] * 4 + [Req(480, 4, 1)] * 4,
         [Req(520, 4, 0)] * 2 + [Req(440, 4, 0)] * 2
         + [Req(520, 4, 1)] * 2 + [Req(440, 4, 1)] * 2),
        ("H46", "turnover_repeat",
         [Req(med, 32, None)] * 8,
         [Req(med, 1 if i % 2 == 0 else 63, None) for i in range(8)]),
        ("H47", "cohort_completion_spread_repeat",
         [Req(med, 16, 0)] * 4 + [Req(med, 16, 1)] * 4,
         [Req(med, 1, 0), Req(med, 31, 0)] * 2
         + [Req(med, 16, 1)] * 4),
        ("H48", "mixed_phase_repeat",
         [Req(long, 1, 0)] * 2 + [Req(long, 1, 1)] * 2
         + [Req(short, 64, 0, delay_s=2.0)] * 2
         + [Req(short, 64, 1, delay_s=2.0)] * 2,
         [Req(short, 64, 0)] * 2 + [Req(short, 64, 1)] * 2
         + [Req(long, 1, 0, delay_s=0.08)] * 2
         + [Req(long, 1, 1, delay_s=0.08)] * 2),
    ]


async def main_async(args) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    images = [
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/astronaut.png",
        "/home/esjung/anaconda3/lib/python3.14/site-packages/skimage/data/coffee.png",
    ]
    all_blocks = blocks(images)
    selected = set(args.hypotheses.split(",")) if args.hypotheses else None
    all_blocks = [x for x in all_blocks if selected is None or x[0] in selected]
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=128)
    rng = random.Random(args.seed)
    rows = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for hyp, name, a, b in all_blocks:
            order = ["A", "B"] * args.pairs
            rng.shuffle(order)
            block_start = time.time()
            pair = 0
            while pair < len(order) or time.time() - block_start < args.min_block_seconds:
                variant = order[pair % len(order)]
                reqs = a if variant == "A" else b
                context = {
                    "hypothesis": hyp, "experiment": name, "variant": variant,
                    "pair_index": pair, "client_epoch": time.time(),
                    "protocol_version": os.environ.get("FLASHVEP_NIGHT_CLIENT_PROTOCOL", "v4_dummy"),
                }
                (root / "active_context.json").write_text(json.dumps(context), encoding="utf-8")
                tasks = [one(session, args.base, args.model, req,
                             f"{hyp}-{variant}-{pair:04d}-{i:03d}", args.seed + pair + i)
                         for i, req in enumerate(reqs)]
                out = await asyncio.gather(*tasks)
                for row in out:
                    row.update(context)
                rows.extend(out)
                with (root / "requests.jsonl").open("a", encoding="utf-8") as fh:
                    for row in out:
                        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                pair += 1
                summary = {
                    "hypothesis": hyp, "variant": variant, "pair": pair,
                    "ok": sum(x["status"] == 200 for x in out),
                    "median_e2e_ms": sorted(x["e2e_ms"] for x in out)[len(out)//2],
                    "elapsed_s": time.time() - block_start,
                }
                print(json.dumps(summary), flush=True)
    (root / "client_complete.json").write_text(json.dumps({
        "requests": len(rows), "hypotheses": [x[0] for x in all_blocks],
        "completed_epoch": time.time(),
    }, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hypotheses", default="")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--min-block-seconds", type=float, default=600)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
