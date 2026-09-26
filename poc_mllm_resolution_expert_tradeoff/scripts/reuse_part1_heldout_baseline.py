"""Materialize an R0/K8 heldout control from the frozen Part-1 EP4 run.

Part 1 completed both public splits under the same checkpoint, processor and
generation contract.  Part 2's 96-request no-op audit established parsed-score
parity for all samples, so the unchanged control can be reused.  Quantities
that Part 1 did not record (notably per-request TTFT) remain null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--part1-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = {
        str(row["request_id"]): row
        for row in json.loads(args.heldout_manifest.read_text(encoding="utf-8"))["requests"]
    }
    source = {}
    for line in args.part1_results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            source[str(row["request_id"])] = row
    missing = sorted(set(requested) - set(source))
    if missing:
        raise RuntimeError(f"Part-1 EP4 output missing heldout IDs: {missing[:5]}")

    records = []
    for base_id, manifest_row in requested.items():
        prior = source[base_id]
        for key in ("image_sha256", "question", "answer", "task"):
            if str(prior.get(key)) != str(manifest_row.get(key)):
                raise RuntimeError(f"Part-1 reuse contract mismatch {base_id} field={key}")
        records.append({
            **prior,
            **{key: value for key, value in manifest_row.items() if key != "request_id"},
            "request_id": f"{base_id}__R0__K8",
            "base_request_id": base_id,
            "resolution_label": "R0",
            "resolution_ratio_target": 1.0,
            "k_visual": 8,
            "k_text": 8,
            "k_decode": 8,
            "actual_visual_token_ratio": 1.0,
            "duplicate_resolution_grid": False,
            "processor_ms": None,
            "ttft_ms": None,
            "request_engine_ms": None,
            "queue_ms": None,
            "evidence": "REUSED_FROZEN_PART1_ACTUAL_EP4_R0_K8_AFTER_96_OF_96_SCORE_PARITY",
            "reuse_limitation": "Part1 did not record request-level TTFT/engine timing",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    temp.replace(args.output)
    atomic_json(args.output.with_suffix(".manifest.json"), {
        "status": "REUSED_VERIFIED_NO_OP_CONTROL",
        "count": len(records),
        "part1_results": str(args.part1_results.resolve()),
        "heldout_manifest": str(args.heldout_manifest.resolve()),
        "no_op_evidence": "Part2 discovery R0/K8 vs Part1 EP4: 96/96 parsed score outcomes",
        "timing_fields": "N/A; use separate paired TRUE-EP4 timing cohort",
    })
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
