"""Keep frozen natural inputs/IDs; change only the common output horizon.

This does not remove graph capture from E2E or invent a warm online scheduler.
It checks whether the observed short-cohort plan ranking is setup amortization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=96)
    args = parser.parse_args()
    assert 1 < args.output_tokens <= 96
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for name in ("b4_c128", "b64_c128"):
        source = args.source / (name + ".jsonl")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        count, context = len(rows), len(rows[0]["input_ids"])
        pages = 2 * count * ((context + 15) // 16)
        pages += count * ((context + args.output_tokens + 15) // 16)
        assert pages < 2048
        for row in rows:
            row["max_new_tokens"] = args.output_tokens
        target = args.out / source.name
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        records.append({"workload": name, "requests": count, "required_pages": pages,
                        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (args.out / "manifest.json").write_text(json.dumps({
        "question": "Does longer actual completion amortize first-use plan/capture ranking?",
        "scope": "fixed natural-content cohort; all setup on target path still charged",
        "only_changed_field": "max_new_tokens", "output_tokens": args.output_tokens,
        "traces": records}, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
