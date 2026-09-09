"""Aggregate paired *request* observer controls; keep transfer scope explicit."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import statistics


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    groups, rows = defaultdict(list), []
    for path in sorted(args.root.glob("analysis_b*/request_summary.csv")):
        for row in csv.DictReader(path.open()):
            row = {key: float(value) if key not in {"family", "bound_scope"} else value for key, value in row.items()}
            row["block"] = path.parent.name
            rows.append(row)
            groups[row["family"]].append(row)
    aggregate = []
    for family, group in sorted(groups.items()):
        overhead = [r["observer_overhead_pct"] for r in group]
        bounds = [r["zero_entire_ttft_e2e_bound_pct"] for r in group]
        aggregate.append({"family": family, "restart_pairs": len(group),
                          "median_observer_overhead_pct": statistics.median(overhead),
                          "min_observer_overhead_pct": min(overhead), "max_observer_overhead_pct": max(overhead),
                          "clean_mean_e2e_median_s": statistics.median(r["clean_mean_e2e_s"] for r in group),
                          "clean_ttft_median_s": statistics.median(r["clean_mean_ttft_s"] for r in group),
                          "zero_entire_ttft_fixed_timeline_bound_pct": statistics.median(bounds),
                          "bound_scope": "prefill-only fixed timeline, not native online queue or total scheduling oracle"})
    with (args.root/"observer_restart_summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.root/"observer_restart_summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
