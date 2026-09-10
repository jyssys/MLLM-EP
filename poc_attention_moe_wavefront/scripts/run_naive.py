#!/usr/bin/env python3
"""Run matched N0 stock or N2 naive concurrent one-third split."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from poc_flashvep.live_causal_modality_wavefront.run_mode import _port, _run_rank


REQUESTS = ("coffee", "method", "coffee_rocket")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stock", "wavefront"), required=True)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.previous / "workload_manifest.json").read_text())
    items = {pair["vision"]["request_id"]: pair["vision"] for pair in manifest["pairs"]}
    rows = []
    for request_id in REQUESTS:
        with np.load(args.previous / items[request_id]["route_file"]) as archive:
            total = int(len(archive["prompt_token_ids"]))
        base = {"request_id": request_id, "prompt_tokens": total,
                "prefix_tokens": round(total / 3), "tail_tokens": total - round(total / 3)}
        for iteration in range(args.warmups):
            rows.append(base | {"phase": "warmup", "measured": False, "iteration": iteration, "timeline": False})
        for iteration in range(args.iterations):
            rows.append(base | {"phase": "measured", "measured": True, "iteration": iteration, "timeline": iteration == 0})
        rows.append(base | {"phase": "correctness", "measured": False, "iteration": 0, "timeline": False})
    for wave, row in enumerate(rows): row["wave"] = wave
    (args.output_dir / "schedule.json").write_text(json.dumps(rows, indent=2) + "\n")
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = _port()
    processes = [context.Process(target=_run_rank, args=(rank, port, args, barrier, rows)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    exits = [process.exitcode for process in processes]
    if exits != [0, 0]: raise RuntimeError(f"{args.mode} failed: {exits}")


if __name__ == "__main__": main()

