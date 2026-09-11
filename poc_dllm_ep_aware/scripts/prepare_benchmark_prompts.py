#!/usr/bin/env python3
"""Materialize deterministic GSM8K/HumanEval anchors from local HF cache."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def read_arrow(path):
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all().to_pylist()


root = Path.home() / ".cache/huggingface/datasets/openai___gsm8k/main/0.0.0"
gsm_path = next(root.glob("*/gsm8k-test.arrow"))
gsm = read_arrow(gsm_path)[:8]
rows = []
for index, item in enumerate(gsm):
    answer = item["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({
        "id": f"gsm8k_{index}",
        "dataset": "GSM8K",
        "prompt": item["question"] + "\nSolve step by step and put the final numeric answer after ####.",
        "reference": answer,
    })

# These are official HumanEval tasks 0--3, embedded only as task prompts; the
# canonical tests remain external and are not copied into results.
human = [
    ("HumanEval/0", "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\"Check whether any two numbers are closer than threshold.\"\"\""),
    ("HumanEval/1", "def separate_paren_groups(paren_string: str) -> list[str]:\n    \"\"\"Split balanced, possibly space-separated parenthesis groups.\"\"\""),
    ("HumanEval/2", "def truncate_number(number: float) -> float:\n    \"\"\"Return the fractional part of a positive float.\"\"\""),
    ("HumanEval/3", "def below_zero(operations: list[int]) -> bool:\n    \"\"\"Return whether a running account balance ever goes below zero.\"\"\""),
]
for task_id, prompt in human:
    rows.append({"id": task_id.replace("/", "_"), "dataset": "HumanEval",
                 "prompt": prompt, "reference": None})

output = Path("poc_dllm_ep_aware/configs/benchmark_prompts.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(output, len(rows))
