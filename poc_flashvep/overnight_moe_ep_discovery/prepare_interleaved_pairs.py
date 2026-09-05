"""Prepare repeated, shuffled F1/F4 pairs for order-confound validation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prepare_regime_grid import build_case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--m-values", default="128,512")
    ap.add_argument("--active-values", default="16")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    ms = [int(x) for x in args.m_values.split(",") if x]
    actives = [int(x) for x in args.active_values.split(",") if x]
    cases = []
    for rep in range(args.reps):
        for m in ms:
            for active in actives:
                for fanout in (1, 4):
                    c = build_case(m, fanout, active, args.layer)
                    c["base_case_id"] = c["case_id"]
                    c["case_id"] = f"{c['case_id']}_R{rep:02d}"
                    c["rep"] = rep
                    cases.append(c)
    random.Random(args.seed).shuffle(cases)
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H1_exact_M_fanout_sign_flip_interleaved",
        "M_values": ms, "fanout_values": [1, 4], "active_values": actives,
        "repetitions_per_condition": args.reps, "random_seed": args.seed,
        "order": "all repeated measurements deterministically shuffled",
        "warmup_protocol": "global model warmup plus per-shape warmups in replay",
        "invariants": ["M", "top_k", "total_assignments", "rank_assignments", "activation"],
        "physical_gpus": [1, 2, 3, 4], "layer": args.layer,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "reps": args.reps}, indent=2))


if __name__ == "__main__":
    main()
