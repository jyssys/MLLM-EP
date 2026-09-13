#!/usr/bin/env python3
"""Check that the schedule harness preserves fixed-B output semantics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_concatenated_json(path: Path) -> list[dict]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    cursor = 0
    rows = []
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        value, cursor = decoder.raw_decode(text, cursor)
        rows.append(value)
    return rows


CASES = [
    (
        "gsm8k",
        32,
        ROOT / "results/clean/replicate_gsm8k_n32_B32_m32_g128_t0.9_L256_r6",
        ROOT / "results/schedules/schedule_dynamic_core_gsm8k_n32_m32_L256_t0.9_r1/policy_fixed32.json",
    ),
    (
        "gsm8k",
        64,
        ROOT / "results/clean/replicate_gsm8k_n32_B64_m32_g128_t0.9_L256_r6",
        ROOT / "results/schedules/schedule_dynamic_core_gsm8k_n32_m32_L256_t0.9_r1/policy_fixed64.json",
    ),
    (
        "humaneval",
        32,
        ROOT / "results/clean/replicate_humaneval_n32_B32_m16_g128_t0.9_L384_r6",
        ROOT / "results/schedules/schedule_dynamic_core_humaneval_n32_m16_L384_t0.9_r1/policy_fixed32.json",
    ),
    (
        "humaneval",
        64,
        ROOT / "results/clean/replicate_humaneval_n32_B64_m16_g128_t0.9_L384_r6",
        ROOT / "results/schedules/schedule_dynamic_core_humaneval_n32_m16_L384_t0.9_r1/policy_fixed64.json",
    ),
]


def main() -> None:
    output = []
    for task, block, static_dir, schedule_path in CASES:
        static_file = next(static_dir.glob("*.jsonl"))
        static = {int(row["id"]): row for row in read_concatenated_json(static_file)}
        scheduled_payload = json.loads(schedule_path.read_text())
        scheduled = {int(row["original_index"]): row for row in scheduled_payload["rows"]}
        common = sorted(set(static) & set(scheduled))
        exact = sum(static[index]["answer"] == scheduled[index]["answer"] for index in common)
        generated_lengths = sum(
            int(static[index]["generated_length"]) == int(scheduled[index]["generated_length"])
            for index in common
        )
        output.append(
            {
                "task": task,
                "block_length": block,
                "requests": len(common),
                "answer_string_exact": exact,
                "answer_string_exact_rate": exact / len(common),
                "generated_length_exact": generated_lengths,
                "generated_length_exact_rate": generated_lengths / len(common),
                "static_source": str(static_dir.relative_to(ROOT.parent)),
                "schedule_source": str(schedule_path.relative_to(ROOT.parent)),
                "interpretation": "fixed-schedule negative control; exact answer equality",
            }
        )
    with (ROOT / "CORRECTNESS_COMPARISON.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
