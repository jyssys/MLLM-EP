from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import WINDOWS, atomic_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.compact)
    violations = []
    maxima = []
    for row in rows:
        observed = sorted({
            int(step)
            for window in WINDOWS
            if window != "1-256"
            for step in row["decode_windows"][window]["observed_steps"]
        })
        maximum = max(observed, default=0)
        maxima.append(maximum)
        if maximum > max(int(row["output_token_count"]) - 1, 0):
            violations.append({
                "request_id": row["request_id"],
                "max_observed_step": maximum,
                "output_token_count": row["output_token_count"],
            })
    atomic_json(args.output, {
        "status": "PASS" if not violations else "FAIL",
        "requests": len(rows),
        "violations": violations,
        "max_sampled_decode_step": max(maxima, default=0),
        "contract": "unique M-RoPE decode position; repeated external-DP padding forwards excluded",
    })
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
