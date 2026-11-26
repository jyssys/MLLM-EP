"""Summarize paired Natural EOS Native cohorts from one server start."""

import json


def requests(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    by_id = {row["request_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Native A/A requires one appearance of each frozen request")
    return by_id


def compare_request_files(first, second):
    a, b = (requests(path) for path in (first, second))
    if a.keys() != b.keys():
        raise ValueError("Native A/A cohorts have different request sets")
    results = []
    for request_id, left in sorted(a.items()):
        right = b[request_id]
        if left["status"] != "MEASURED_ONLINE" or right["status"] != "MEASURED_ONLINE":
            raise RuntimeError(f"Native A/A request did not complete: {request_id}")
        x, y = left["token_ids"], right["token_ids"]
        first_difference = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        results.append({"request_id": request_id, "exact": x == y, "output_tokens": [len(x), len(y)],
                        "first_distinct_token": first_difference if x != y else None})
    return results


def compare_online_native(run, first, second, metrics):
    raw = run / "raw"
    results = compare_request_files(raw / f"{first}_requests.jsonl", raw / f"{second}_requests.jsonl")
    summary = {"tags": [first, second], "matched_exact_n": sum(row["exact"] for row in results), "n": len(results),
               "evidence": "MEASURED_ONLINE", "scope": "One server start, two Natural EOS Native cohorts",
               "cohort_metrics": metrics, "requests": results}
    output = run / "analysis" / f"{first}_comparison.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("NATIVE_ONLINE_AA", first, summary["matched_exact_n"], summary["n"], flush=True)
