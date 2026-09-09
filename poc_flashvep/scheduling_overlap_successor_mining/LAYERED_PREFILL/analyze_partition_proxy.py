"""Native measured-layer minimax diagnostic, explicitly not a request oracle."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "COMMON"))
from analyze_transfer_cpu import balanced_partition, save_csv


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    args = parser.parse_args()
    rows = []
    incomplete = []
    for path in sorted(args.screen.glob("*/layer_profiles.csv")):
        profiles = defaultdict(dict)
        for row in csv.DictReader(path.open()):
            key = row["rank"], row["phase"], row["prefill_active"]
            layer = int(row["layer"].rsplit(".", 1)[-1])
            assert layer not in profiles[key]
            profiles[key][layer] = float(row["layer_median_ms"])
        for key, profile in profiles.items():
            if set(profile) != set(range(48)):
                incomplete.append({"run": path.parent.name, "condition": key,
                                   "layers": len(profile)})
                continue
            costs = [profile[i] for i in range(48)]
            for groups in (4, 12, 16, 24):
                width = 48 // groups
                equal = max(sum(costs[i:i+width]) for i in range(0, 48, width))
                optimum, cuts = balanced_partition(costs, groups)
                rows.append({"run": path.parent.name, "rank": key[0],
                             "phase": key[1], "prefill_active": key[2],
                             "groups": groups, "equal_group_ms": equal,
                             "minimax_group_ms": optimum,
                             "group_proxy_reduction_pct": 100*(1-optimum/equal),
                             "cuts": json.dumps(cuts),
                             "direct_request_e2e_gain": "NOT_ESTABLISHED"})
    save_csv(args.screen / "native_partition_proxy.csv", rows)
    summary = {"profiles": len(rows)//4, "incomplete": incomplete,
               "scope": "measured eager unsampled per-layer medians, not graph performance",
               "limitations": ["different group partition can change batch/expert/cache costs",
                               "minimax groups do not model request progress or arrivals",
                               "rank-local summaries, no cross-device timestamp subtraction",
                               "not evidence of direct E2E successor headroom"]}
    (args.screen / "native_partition_proxy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
