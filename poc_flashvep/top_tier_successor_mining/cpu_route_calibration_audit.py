"""CPU-only SERE calibration-arithmetic sensitivity on existing real route IDs.

Subsampled prefill populations are explicitly NOT captured decode states or new
GPU evidence. This measures mapping sensitivity, never benchmark-quality impact.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from policies import sere_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--requests", type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(4)
    root = args.results
    out = root / "analysis/cpu_release_checkpoint"
    out.mkdir(parents=True, exist_ok=True)
    old = torch.load(root / "quality/sere_fineweb_400x128/similarity.pt", map_location="cpu", weights_only=True)
    new = torch.load(root / "quality/sere_fineweb_400x128_official_norm/similarity.pt", map_location="cpu", weights_only=True)
    routes = root / "quality/modes_calib1024_official/routes"
    requests = sorted(p.name.removesuffix("_prefill_l00.pt") for p in routes.glob("*_prefill_l00.pt"))[:args.requests]
    rows = []
    for index, request in enumerate(requests):
        for layer in (0, 1, 2, 3, 12, 24, 36, 47):
            record = torch.load(routes / f"{request}_prefill_l{layer:02d}.pt", map_location="cpu", weights_only=True)
            for modality in ("vision", "text", "all"):
                candidates = torch.arange(len(record["ids"])) if modality == "all" else record[modality].nonzero().flatten()
                for budget in (1, 4, 16, 128):
                    if len(candidates) < budget:
                        continue
                    rng = np.random.default_rng(7317 + index*100 + layer)
                    positions = rng.choice(candidates.numpy(), budget, replace=False)
                    ids = record["ids"][positions]
                    weights = record["weights"][positions].float()
                    for retain, threshold in ((2, .5), (4, .5), (2, .7)):
                        previous = sere_route(ids, old[layer], retain, threshold)
                        official = sere_route(ids, new[layer], retain, threshold)
                        different = previous != official
                        rows.append({"request": request, "layer": layer, "modality": modality,
                            "sampled_prefill_tokens": budget, "retain": retain, "rho": threshold,
                            "assignments": ids.numel(), "different_assignments": int(different.sum()),
                            "different_router_mass": float(weights[different].sum()),
                            "total_router_mass": float(weights.sum()),
                            "old_changed_assignments": int((previous != ids).sum()),
                            "official_changed_assignments": int((official != ids).sum()),
                            "scope": "CPU_SUBSAMPLED_PREFILL_ROUTE_MAPPING_NOT_DECODE_OR_QUALITY"})
        if index % 6 == 0:
            print(json.dumps({"requests_completed": index+1, "conditions": len(rows)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "sere_calibration_route_sensitivity.csv", index=False)
    summary = []
    for keys, group in frame.groupby(["retain", "rho", "sampled_prefill_tokens", "modality"]):
        summary.append({"retain": int(keys[0]), "rho": keys[1], "sampled_prefill_tokens": int(keys[2]),
                        "modality": keys[3], "conditions": len(group),
                        "assignment_difference_percent": 100*group.different_assignments.sum()/group.assignments.sum(),
                        "weighted_mass_difference_percent": 100*group.different_router_mass.sum()/group.total_router_mass.sum(),
                        "old_changed_percent": 100*group.old_changed_assignments.sum()/group.assignments.sum(),
                        "official_changed_percent": 100*group.official_changed_assignments.sum()/group.assignments.sum()})
    pd.DataFrame(summary).to_csv(out / "sere_calibration_route_sensitivity_summary.csv", index=False)
    print(json.dumps({"conditions": len(frame), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
