#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_rank_aware_active_expert_reduction.contract import write_jsonl_gz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subdir", default="raw_stage")
    args = parser.parse_args()
    rows = []
    for path in sorted((args.run / args.subdir).glob("worker_pid*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    write_jsonl_gz(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
