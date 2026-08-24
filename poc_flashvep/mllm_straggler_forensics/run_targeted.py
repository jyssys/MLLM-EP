"""Run only the preregistered fast/slow live pair for replay or Nsight."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace

from poc_flashvep.live_prefill_execution_regime.run_live import _port, _run_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("replay", "profile"), required=True)
    parser.add_argument("--profile-repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())["selected_cross_modality_pair"]
    prior_schedule = json.loads((args.source / "schedule.json").read_text())
    source_dp = {}
    pair_metadata = {}
    for row in prior_schedule:
        source_dp[row["request_id"]] = int(row["source_dp_rank"])
        pair_metadata[row["request_id"]] = row
    schedule = []
    repeats = 1 if args.mode == "replay" else args.profile_repeats
    for iteration in range(repeats):
        for modality, prefix in (("text", "text"), ("vision", "vision")):
            request_id = selection[f"{prefix}_request_id"]
            meta = pair_metadata[request_id]
            schedule.append({
                "request_id": request_id, "modality": modality,
                "pair_id": int(meta["pair_id"]), "token_bucket": meta["token_bucket"],
                "prompt_tokens": int(meta["prompt_tokens"]), "source_dp_rank": source_dp[request_id],
                "phase": args.mode, "instrument": False, "measured": True,
                "iteration": iteration, "target": True,
                "layer": int(selection["layer"]), "rank": int(selection["rank"]),
            })
    for wave, row in enumerate(schedule):
        row["wave"] = wave
    (args.output_dir / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    os.environ["FLASHVEP_FORENSIC_CONTROL"] = str((args.output_dir / "control.json").resolve())
    os.environ["FLASHVEP_FORENSIC_OUTPUT"] = str((args.output_dir / "target_results").resolve())
    os.environ["FLASHVEP_FORENSIC_MODE"] = args.mode
    runtime_args = SimpleNamespace(output_dir=args.output_dir, previous=args.source, model_path=args.model_path)
    context = mp.get_context("spawn"); barrier = context.Barrier(2); port = _port()
    processes = [context.Process(target=_run_rank, args=(rank, port, runtime_args, barrier, schedule)) for rank in range(2)]
    for process in processes: process.start()
    for process in processes: process.join()
    codes = [process.exitcode for process in processes]
    if codes != [0, 0]:
        raise RuntimeError(f"targeted run failed: {codes}")


if __name__ == "__main__":
    main()
