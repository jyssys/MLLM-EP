"""Prepare repeated shuffled uniform/skew distribution-shape pairs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prepare_distribution_grid import build_case


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m-values", default="128,512"); ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--layer", type=int, default=24); ap.add_argument("--seed", type=int, default=616)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    ms = [int(x) for x in args.m_values.split(",") if x]
    cases = []
    for rep in range(args.reps):
        for m in ms:
            for shape in ("uniform", "skew"):
                c = build_case(m, shape, args.layer); c["case_id"] += f"_R{rep:02d}"; c["rep"] = rep; cases.append(c)
    random.Random(args.seed).shuffle(cases)
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H7_distribution_shape_interleaved", "M_values": ms,
        "shapes": ["uniform", "skew"], "reps": args.reps, "seed": args.seed,
        "invariants": ["M", "top_k", "total_assignments", "rank_assignments", "active_experts", "activation"],
        "physical_gpus": [1, 2, 3, 4], "layer": args.layer,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__": main()
