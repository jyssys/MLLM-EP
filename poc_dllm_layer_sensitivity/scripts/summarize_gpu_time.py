#!/usr/bin/env python3
import csv
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source = root / "GPU_TIME_LOG.raw.csv"
rows = []
with source.open() as handle:
    for values in csv.reader(handle):
        if not values:
            continue
        start, end, wall, gpu_set, kind, dataset, configuration, log = values
        rows.append(
            {
                "start_epoch": start,
                "end_epoch": end,
                "wall_seconds": wall,
                "physical_gpu_set": gpu_set,
                "gpu_count": 4,
                "gpu_hours": float(wall) * 4.0 / 3600.0,
                "kind": kind,
                "dataset": dataset,
                "configuration": configuration,
                "log": log,
            }
        )
with (root / "GPU_TIME_LOG.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(
    {
        "runs": len(rows),
        "live_gpu_wall_hours": sum(float(row["wall_seconds"]) for row in rows) / 3600,
        "gpu_hours": sum(float(row["gpu_hours"]) for row in rows),
    }
)
