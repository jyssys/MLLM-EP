"""Append explicitly scoped experiment wall time; never infer CUDA active time."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--log", type=Path, required=True)
p.add_argument("--run-id", required=True)
p.add_argument("--candidate", required=True)
p.add_argument("--question", required=True)
p.add_argument("--start", type=float, required=True)
p.add_argument("--end", type=float, required=True)
p.add_argument("--gpus", default="4,5,6,7")
p.add_argument("--kind", required=True,
               choices=["correctness_diagnostic", "measurement", "mechanism_diagnostic"])
p.add_argument("--valid", required=True)
p.add_argument("--artifact", required=True)
args = p.parse_args()
gpus = [int(x) for x in args.gpus.split(",")]
assert set(gpus) <= {4, 5, 6, 7} and len(gpus) == len(set(gpus))
assert args.end >= args.start
with args.log.open() as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    assert not any(row["run_id"] == args.run_id for row in reader), "Duplicate interval"
duration = args.end-args.start
row = dict(run_id=args.run_id, candidate=args.candidate,
           research_question=args.question,
           start_utc=datetime.fromtimestamp(args.start, timezone.utc).isoformat(),
           end_utc=datetime.fromtimestamp(args.end, timezone.utc).isoformat(),
           physical_gpus=args.gpus, elapsed_seconds=duration,
           gpu_hours=duration*len(gpus)/3600,
           kind=args.kind, valid=args.valid, artifact=args.artifact)
with args.log.open("a") as handle:
    csv.DictWriter(handle, fieldnames=fields).writerow(row)
print(row)
