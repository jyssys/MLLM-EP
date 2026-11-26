"""Measure split/native co-batching from observed prepared D request identities."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from ..mllm_epd_h200_prefill_heavy_batch_matched.analysis import jsonl, read


D_ID = re.compile(r"chatcmpl-(.+):D(?:-[A-Za-z0-9]{8})?")


def runtime_id(identifier):
    match = D_ID.fullmatch(identifier)
    if match is None:
        raise ValueError(f"Unknown D request identifier {identifier}")
    return match[1]


def observe(run, cohort, suffixes):
    receipts = read(run / "analysis" / f"{cohort}_batch.json")
    receipt, = receipts
    if not receipt["valid"]:
        raise ValueError(f"Invalid batch counter for {run.name}/{cohort}")
    source = Path(receipt["folder"]) / "decode_identities.jsonl"
    if not source.exists():
        return None
    expected = {key: count for key, count in suffixes.items() if key.startswith(cohort + "_")}
    prompt_rows = Counter()
    distribution = Counter()
    split_steps = shared_steps = total_decode_rows = token_weighted = 0
    host_seconds = host_weighted = 0.0
    for step in jsonl(source):
        decoded = []
        for row in step["requests"]:
            identifier = runtime_id(row["request_id"])
            prompt_rows[identifier] += row["prompt_rows"]
            if row["decode_rows"]:
                decoded.append((identifier, row["decode_rows"]))
        if not decoded:
            continue
        count = len(decoded)
        distribution[count] += 1
        rows = sum(rows for _, rows in decoded)
        total_decode_rows += rows
        token_weighted += rows * count
        elapsed = step["end"] - step["begin"]
        host_seconds += elapsed
        host_weighted += elapsed * count
        with_split = sum(identifier in expected for identifier, _ in decoded)
        split_steps += with_split > 0
        shared_steps += 0 < with_split < count
    decode_steps = sum(distribution.values())
    gate_path = run / "analysis" / f"{cohort.removesuffix('_observed')}_batch_qualification.json"
    gate = read(gate_path) if gate_path.exists() else None
    return {
        "start": run.name,
        "cohort": cohort,
        "evidence": "DIAGNOSTIC" if not gate or not gate["passed"] else "TRACE_ANALYSIS",
        "prepared_decode_steps": decode_steps,
        "prepared_batch_histogram": dict(sorted(distribution.items())),
        "step_weighted_mean_decode_batch": sum(k * v for k, v in distribution.items()) / decode_steps
        if decode_steps else None,
        "token_weighted_mean_decode_batch": token_weighted / total_decode_rows if total_decode_rows else None,
        "host_forward_enclosure_weighted_mean": host_weighted / host_seconds if host_seconds else None,
        "split_decode_steps": split_steps,
        "native_split_co_batch_steps": shared_steps,
        "native_split_co_batch_fraction_of_split_steps": shared_steps / split_steps if split_steps else None,
        "suffix_query_coverage": {key: {"actual_D_prompt_rows": prompt_rows[key], "expected": count,
                                      "exact": count == prompt_rows[key]} for key, count in expected.items()},
        "scope": "CPU input preparation and host forward enqueue enclosures from a separate observed cohort.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        suffixes = {}
        for item in jsonl(run / "raw" / "proxy_epd.jsonl"):
            suffix = (item.get("token_partition") or {}).get("suffix_tokens", 0)
            if suffix:
                suffixes[item["runtime_request_id"]] = suffix
        for path in sorted((run / "analysis").glob("*_batch.json")):
            result = observe(run, path.stem.removesuffix("_batch"), suffixes)
            if result is not None:
                results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
