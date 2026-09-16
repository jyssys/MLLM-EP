"""Score official HumanEval checks in an isolated user/mount/network namespace."""

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path

from score_reference import wilson


HERE = Path(__file__).resolve().parent


def candidate_from_output(record):
    text = record["postprocessed_generation"].strip()
    entry = record["entry_point"]
    entry_pattern = r"^[ \t]*(?:async[ \t]+)?def[ \t]+" + re.escape(entry) + r"[ \t]*\("
    fences = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", text, flags=re.I | re.S)
    solutions = [fence for fence in fences if re.search(entry_pattern, fence, flags=re.M)]
    if solutions:
        # Explanations can contain example fences before the actual solution.
        # Select the final fenced implementation of the official entry point.
        text = solutions[-1].strip()
    elif fences:
        text = fences[0].strip()
    if re.search(entry_pattern, text, flags=re.M):
        # A complete replacement still needs the official prompt's import /
        # helper prelude. Do not append the original unfinished target function:
        # that would define the entry point twice, while dropping the prelude
        # causes false NameError failures for annotations such as List[int].
        prompt = record["original_prompt"]
        target = re.search(r"^[ \t]*(?:async[ \t]+)?def[ \t]+" + re.escape(entry) + r"[ \t]*\(",
                           prompt, flags=re.M)
        prelude = prompt[:target.start()] if target else ""
        return prelude + text, "complete-function"
    return record["original_prompt"] + text, "prompt-continuation"


def sandbox_check(record):
    candidate, mode = candidate_from_output(record)
    payload = {"candidate": candidate, "test": record["test"], "entry_point": record["entry_point"]}
    try:
        process = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--net", "--mount", "--pid", "--fork", "--kill-child",
             "bash", str(HERE / "sandbox_humaneval.sh"), str(HERE / "humaneval_sandbox_runner.py")],
            input=json.dumps(payload), text=True, capture_output=True, timeout=8, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "failure": "TIMEOUT", "candidate_mode": mode}
    if process.returncode or not process.stdout.strip():
        return {"passed": False, "failure": "SANDBOX_FAILURE", "candidate_mode": mode,
                "detail": process.stderr[-300:]}
    result = json.loads(process.stdout.strip().splitlines()[-1])
    result["candidate_mode"] = mode
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sample-id-max", type=int)
    args = parser.parse_args()
    records = [json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda row: row["sample_id"])
    if args.sample_id_max is not None:
        records = [row for row in records if row["sample_id"] <= args.sample_id_max]
    if not records or any(row["task"] != "humaneval" for row in records):
        raise RuntimeError("expected captured HumanEval samples")
    if len({row["sample_id"] for row in records}) != len(records):
        raise RuntimeError("duplicate HumanEval sample IDs")
    for row in records:
        result = sandbox_check(row)
        row["humaneval_pass"] = result["passed"]
        row["humaneval_failure"] = result["failure"]
        row["candidate_mode"] = result["candidate_mode"]
        row["sandbox_detail"] = result.get("detail")
        print(f"HumanEval id={row['sample_id']} pass={result['passed']} failure={result['failure']}", flush=True)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.samples.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    passed = sum(row["humaneval_pass"] for row in records)
    summary = {
        "task": "humaneval", "n": len(records), "passed": passed,
        "pass_at_1": passed / len(records), "wilson_95": wilson(passed, len(records)),
        "mean_nfe": statistics.mean(row["nfe"] for row in records),
        "median_nfe": statistics.median(row["nfe"] for row in records),
        "syntax_failures": sum(row["humaneval_failure"] == "SYNTAX_ERROR" for row in records),
        "sandbox_failures": sum(row["humaneval_failure"] == "SANDBOX_FAILURE" for row in records),
        "truncated": sum(row["termination_reason"] == "GEN_LENGTH_CAP" for row in records),
        "empty_output": sum(not row["postprocessed_generation"].strip() for row in records),
        "failure_counts": {name: sum(row["humaneval_failure"] == name for row in records)
                           for name in sorted({row["humaneval_failure"] for row in records if row["humaneval_failure"]})},
        "protocol_caveat": "Generated chat output is evaluated as complete function if it defines the entry point, otherwise as official prompt continuation. This differs from some published prompting protocols.",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
