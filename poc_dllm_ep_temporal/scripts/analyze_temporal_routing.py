#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_dllm_ep_temporal.dllm_ep.routing_metrics import hot_expert_persistence, rank_temporal_summary
from poc_dllm_ep_temporal.dllm_ep.trace_schema import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = read_jsonl(args.trace)
    summary = {
        "records": len(records),
        "requests": len({item.request_id for item in records}),
        "rank_temporal": rank_temporal_summary(records),
        "hot_expert_temporal": hot_expert_persistence(records),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
