from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import HISTORICAL_ROUTE_TRACE, MODEL
from .contract import audit_environment, parse_physical_gpus
from .evaluation.registry import registry
from .io import atomic_json, completed_ids
from .virtual_ep.cost import validate
from .virtual_ep.replay import write_replay


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mllm-ep-foundation")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit_environment")
    audit.add_argument("--physical-gpus", default="4,5,6,7")
    audit.add_argument("--model", type=Path, default=MODEL)
    audit.add_argument("--output", type=Path, required=True)
    _common(audit)

    replay = sub.add_parser("virtual_replay")
    replay.add_argument("--physical-gpus", default="4")
    replay.add_argument("--target-ep", type=int, choices=(2, 4, 8), required=True)
    replay.add_argument("--target-topology", default="tp2_dp2_source_shards")
    replay.add_argument("--trace", type=Path, default=HISTORICAL_ROUTE_TRACE)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--waves", help="comma-separated wave IDs")
    _common(replay)

    calibration = sub.add_parser("calibrate_cost")
    calibration.add_argument("--backend", required=True)
    calibration.add_argument("--shape-manifest", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    _common(calibration)

    sim = sub.add_parser("validate_sim")
    sim.add_argument("--calibration", type=Path, required=True)
    sim.add_argument("--holdout-manifest", type=Path, required=True)
    sim.add_argument("--output", type=Path, required=True)
    _common(sim)

    resume = sub.add_parser("resume_experiment")
    resume.add_argument("--physical-gpus", default="4,5")
    resume.add_argument("--completed-index", type=Path, required=True)
    resume.add_argument("--request-manifest", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    _common(resume)

    baseline = sub.add_parser("evaluate_baseline")
    baseline.add_argument("--physical-gpus", default="4,5,6,7")
    baseline.add_argument("--lane", choices=("official", "deterministic"), required=True)
    baseline.add_argument("--backend", choices=("reference", "physical_ep"), required=True)
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    _common(baseline)

    anchor = sub.add_parser("capture_anchor")
    anchor.add_argument("--physical-gpus", default="4,5,6,7")
    anchor.add_argument("--manifest", type=Path, required=True)
    anchor.add_argument("--output", type=Path, required=True)
    _common(anchor)

    args = parser.parse_args()
    if args.command == "audit_environment":
        result = audit_environment(args.output, args.physical_gpus, args.model, args.dry_run)
    elif args.command == "virtual_replay":
        parse_physical_gpus(args.physical_gpus)
        waves = None if not args.waves else [int(item) for item in args.waves.split(",")]
        result = ({"command": args.command, "physical_gpus": args.physical_gpus,
                   "target_ep": args.target_ep, "trace": str(args.trace),
                   "dry_run": True} if args.dry_run else
                  write_replay(args.output, args.trace, args.target_ep, waves))
    elif args.command == "calibrate_cost":
        shapes = json.loads(args.shape_manifest.read_text())
        result = {
            "schema_version": "virtual_ep_cost.v1",
            "backend": args.backend,
            "signature": shapes.get("signature"),
            "observations": shapes.get("observations", []),
            "heldout_truth": shapes.get("heldout_truth", {}),
            "split": shapes.get("split"),
            "status": "READY" if shapes.get("observations") else "INCOMPLETE_NO_OBSERVATIONS",
        }
        if not args.dry_run:
            atomic_json(args.output, result)
    elif args.command == "validate_sim":
        result = ({"command": args.command, "dry_run": True} if args.dry_run else
                  validate(args.calibration, args.holdout_manifest, args.output))
    elif args.command == "resume_experiment":
        physical = parse_physical_gpus(args.physical_gpus)
        manifest = json.loads(args.request_manifest.read_text())
        done = completed_ids(args.completed_index)
        ids = [str(row["request_id"]) for row in manifest["requests"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate request IDs in manifest")
        result = {"physical_gpus": physical, "completed": sorted(done),
                  "pending": [value for value in ids if value not in done],
                  "missing_completed_ids": sorted(done - set(ids)),
                  "status": "READY_TO_RESUME"}
        if not args.dry_run:
            atomic_json(args.output, result)
    elif args.command in ("evaluate_baseline", "capture_anchor"):
        parse_physical_gpus(args.physical_gpus)
        result = {
            "status": "DRY_RUN" if args.dry_run else "INCOMPLETE_REQUIRES_GPU_RUNNER",
            "command": args.command,
            "physical_gpus": args.physical_gpus,
            "lane": getattr(args, "lane", None),
            "backend": getattr(args, "backend", None),
            "manifest": str(args.manifest.resolve()),
            "note": "Use scripts/run_full_model.py; this command never fabricates a completed GPU run.",
        }
        if not args.dry_run:
            atomic_json(args.output, result)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    printable = result
    if args.command == "virtual_replay" and not args.dry_run:
        rows = result.get("rows", [])
        printable = {
            key: value for key, value in result.items() if key != "rows"
        }
        printable["row_count"] = len(rows)
        printable["request_count"] = len({row["request_id"] for row in rows})
        printable["output"] = str(args.output.resolve())
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
