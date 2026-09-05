"""Convert captured Qwen3-VL routing archives into bounded replay cases.

The routes are taken verbatim from a prior real-image Qwen3-VL capture.  The
validated layer-24 hidden activation is used by the replay hook only as a
shape-compatible input, so results are explicitly labelled route-transfer
diagnostics rather than an end-to-end model rerun.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--m-values", default="128,512")
    ap.add_argument("--layer", type=int, default=24); args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest = json.loads((args.source / "sample_manifest.json").read_text())
    by_id = {x["sample_id"]: x for x in manifest["samples"]}
    ms = [int(x) for x in args.m_values.split(",") if x]
    cases = []
    selected = []
    # One natural/fine-grained/chart sample per available size, plus the
    # multi-image sample when present in the source capture.
    for sid in ("astronaut", "motorcycle", "deep_field", "retina", "model_card", "method", "coffee_rocket"):
        paths = glob.glob(str(args.source / f"routing.dp*.{sid}.npz"))
        if not paths or sid not in by_id: continue
        selected.append(sid)
        archive = np.load(paths[0]); routes_all = archive["routed_experts"]
        for m in ms:
            if routes_all.shape[0] < m: continue
            routes = routes_all[:m, args.layer, :].astype(np.int64)
            rank = (routes // 32).reshape(-1)
            fanout = np.asarray([len(np.unique(row // 32)) for row in routes], dtype=np.float64)
            counts = np.bincount(routes.reshape(-1), minlength=128)
            rank_counts = np.bincount(rank, minlength=4)
            active = int(np.count_nonzero(counts))
            c = by_id[sid]
            cases.append({
                "case_id": f"real_{sid}_M{m}_F{fanout.mean():.2f}_A{active}",
                "request_id": sid, "category": c.get("category", "real"), "modality": "real_qwen3vl_route",
                "layer": args.layer, "M": m, "routes": routes.tolist(), "token_count": m,
                "total_assignments": int(routes.size), "fanout_ranks_mean": float(fanout.mean()),
                "fanout_ranks_median": float(np.median(fanout)), "fanout_histogram": {str(int(k)): int(v) for k,v in zip(*np.unique(fanout, return_counts=True))},
                "active_experts": active, "rank_assignments": rank_counts.astype(int).tolist(),
                "expert_counts": counts.astype(int).tolist(),
                "route_sha256": hashlib.sha256(routes.tobytes()).hexdigest(),
                "image_count": len(c.get("images", [])), "vision_tokens": c.get("processor_vision_tokens"),
            })
    if not cases: raise RuntimeError("no source route cases")
    (args.output / "cases.json").write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    (args.output / "route_statistics.json").write_text(json.dumps({"selected_samples": selected, "cases": cases}, indent=2) + "\n")
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "H8_real_route_transfer", "source": str(args.source), "layer": args.layer,
        "M_values": ms, "route_source": "captured Qwen3-VL real-image top-k IDs",
        "activation_source": "validated BF16 layer-24 capture (cycled by replay)",
        "caveat": "route-transfer operator evidence; not a same-request hidden-state replay",
        "physical_gpus": [1, 2, 3, 4],
    }, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "samples": selected}, indent=2))


if __name__ == "__main__": main()
