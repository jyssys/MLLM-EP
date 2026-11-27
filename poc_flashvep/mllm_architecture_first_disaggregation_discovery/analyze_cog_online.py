"""Summarize matched online runs and a frozen local answer diagnostic."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import numpy as np


EXPLICIT = re.compile(r"(?im)^\s*(?:final answer|answer)\s*[:：]\s*(\S[^\n]*)$")
OPTION = re.compile(r"^\(?([a-z])\)?[.)]?$", re.IGNORECASE)


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().strip(" `*").split())


def answer(text):
    matches = EXPLICIT.findall(text)
    line = matches[-1] if matches else text
    line = re.sub(r"\s*</?(?:answer|final)>\s*", "", line, flags=re.IGNORECASE)
    return normalize(line)


def distance(left, right):
    previous = list(range(len(right) + 1))
    for index, lhs in enumerate(left, 1):
        current = [index]
        for offset, rhs in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (lhs != rhs)))
        previous = current
    return previous[-1]


def score(row, text):
    """Locally score explicit final lines or raw output. Never call an external extractor."""
    prediction = answer(text)
    references = row["reference_answers"]
    if not references or any(not isinstance(item, (str, int, float)) for item in references):
        return None, "reference_unavailable"
    refs = [answer(item) for item in references]
    source = row["source"]
    if source in ("lmms-lab/DocVQA", "lmms-lab/MP-DocVQA"):
        similarities = [1 - distance(prediction, key) / max(len(prediction), len(key), 1) for key in refs]
        best = max(similarities)
        return (best if best > 0.5 else 0.), "local_explicit_line_anls"
    options = row["options"]
    if options and isinstance(options, (list, dict)):
        choices = dict(options) if isinstance(options, dict) else {chr(97+i): v for i, v in enumerate(options)}
        choices = {normalize(key): answer(value) for key, value in choices.items()}
        result = OPTION.fullmatch(prediction)
        if result and result[1].lower() in choices:
            key = result[1].lower()
            return float(key in refs or choices[key] in refs), "local_choice"
        return float(prediction in refs or any(key in refs and prediction == value for key, value in choices.items())), "local_choice"
    return float(prediction in refs), "local_explicit_line_exact"


def quality(rows, output):
    predicted = {}
    counter = defaultdict(list)
    for result in output["requests"]:
        ident = result["request_id"]
        value, kind = score(rows[ident], result["text"])
        predicted[ident] = value
        if value is not None:
            counter[(result["source"], kind)].append(value)
    return predicted, [
        {"source": source, "method": kind, "count": len(values), "mean": sum(values) / len(values)}
        for (source, kind), values in sorted(counter.items())
    ]


def compare(baseline, candidate, refs):
    contract = ("planned_arrivals", "cap", "offered_qps", "configured_slots_per_decode_worker")
    if any(baseline[key] != candidate[key] for key in contract):
        raise ValueError("Online comparison needs matched arrivals, IDs and Natural cap")
    base, new = [dict((row["request_id"], row) for row in run["requests"]) for run in (baseline, candidate)]
    if set(base) != set(new):
        raise ValueError("Incomplete online pair")
    exact = Counter()
    sample = []
    for ident, previous in base.items():
        after = new[ident]
        if any(previous[key] != after[key] for key in ("prompt_sha256", "input_tokens", "source")):
            raise ValueError(f"The candidate changed the original input for {ident}")
        source = previous["source"]
        a, b = previous["tokens"], after["tokens"]
        prefix = next((i for i, (i1, i2) in enumerate(zip(a, b)) if i1 != i2), min(len(a), len(b)))
        exact[source] += int(a == b)
        sample.append({"request_id": ident, "source": source, "exact_tokens": a == b,
                       "common_prefix": prefix, "baseline_tokens": len(a), "candidate_tokens": len(b),
                       "baseline_score": score(refs[ident], previous["text"])[0],
                       "candidate_score": score(refs[ident], after["text"])[0]})
    per_source = Counter(row["source"] for row in base.values())
    return {"offered_qps": candidate["offered_qps"], "natural_exact": sum(exact.values()),
            "count": len(base), "requests": sample,
            "exact_by_source": {key: {"exact": exact[key], "count": value} for key, value in per_source.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = {row["request_id"]: row for row in json.loads(args.manifest.read_text())["requests"]}
    results = [json.loads(path.read_text()) for path in args.runs]
    summaries, pairs = [], []
    for result in results:
        output = result["requests"]
        summary = {key: value for key, value in result.items() if key not in ("requests", "slot_events")}
        summary["input_lengths_median_p95"] = list(np.percentile([row["input_tokens"] for row in output], [50, 95]))
        summary["output_lengths_median_p95"] = list(np.percentile([len(row["tokens"]) for row in output], [50, 95]))
        summary["finish"] = dict(Counter(row["finish"] for row in output))
        summary["local_quality"] = quality(rows, result)[1]
        summaries.append(summary)
        baseline_modes = ("resident_text_fast", "dp2")
        if result["mode"] == "visual_pruned":
            baseline_modes += ("generic_pruned",)
        for baseline in results:
            if (baseline["mode"] in baseline_modes and baseline["mode"] != result["mode"]
                    and baseline["planned_arrivals"] == result["planned_arrivals"]):
                pairs.append({"baseline": baseline["mode"], "candidate": result["mode"], **compare(baseline, result, rows)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump({"scope": "In-process Natural open-loop. Local deterministic scorers have no vendor verification.",
                   "summaries": summaries, "matched_outputs": pairs}, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
