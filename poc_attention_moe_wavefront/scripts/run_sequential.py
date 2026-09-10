#!/usr/bin/env python3
"""Run matched N1 single-owner sequential physical split at one third."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from poc_flashvep.non_dbo_causal_wavefront.run_stage0 import _port, _rank


REQUESTS = ("coffee", "method", "coffee_rocket")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args(); args.variant = "S"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.previous / "workload_manifest.json").read_text())
    items = {pair["vision"]["request_id"]: pair["vision"] for pair in manifest["pairs"]}
    rows = []
    for request_id in REQUESTS:
        with np.load(args.previous / items[request_id]["route_file"]) as archive:
            total = int(len(archive["prompt_token_ids"]))
        base = {"request_id": request_id, "variant": "S", "prompt_tokens": total,
                "prefix_tokens": round(total / 3), "tail_tokens": total - round(total / 3)}
        for iteration in range(args.warmups):
            rows.append(base | {"phase": "warmup", "measured": False, "iteration": iteration,
                                "stage_profile": False, "flush_after": False})
        for iteration in range(args.iterations):
            rows.append(base | {"phase": "measured", "measured": True, "iteration": iteration,
                                "stage_profile": iteration == 0, "flush_after": False})
        rows.append(base | {"phase": "correctness", "measured": False, "iteration": 0,
                            "stage_profile": False, "flush_after": False})
    for wave, row in enumerate(rows): row["wave"] = wave
    rows[-1]["flush_after"] = True
    (args.output_dir / "schedule.json").write_text(json.dumps(rows, indent=2) + "\n")
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = _port()
    processes = [context.Process(target=_rank, args=(rank, port, args, barrier, rows)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]: raise RuntimeError(f"N1 failed: {exits}")


if __name__ == "__main__": main()

