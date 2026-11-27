"""Compare E/P stage coactivity and duration in the matched open-loop trials."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def summary(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def quantile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        return values[lower] + (index - lower) * (values[min(lower + 1, len(values) - 1)] - values[lower])

    return {"n": len(values), "p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99)}


class Intervals:
    def __init__(self, samples):
        merged = []
        for lo, hi in sorted(samples):
            if hi < lo:
                raise ValueError("An interval ended before it began")
            if merged and lo <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(hi, merged[-1][1])
            else:
                merged.append((lo, hi))
        self.samples = merged
        self.starts = [lo for lo, _ in merged]
        self.ends = [hi for _, hi in merged]

    def contains(self, at):
        index = bisect_right(self.starts, at) - 1
        return index >= 0 and at < self.ends[index]

    def overlap(self, lo, hi):
        first = bisect_right(self.ends, lo)
        end = bisect_left(self.starts, hi)
        return sum(min(hi, b) - max(lo, a) for a, b in self.samples[first:end])


def phase_label(e, p):
    return ("neither", "P", "E", "E+P")[2 * bool(e) + bool(p)]


def context_bin(rows):
    for upper in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
        if rows <= upper:
            return upper
    return "more"


def load(path):
    payload = path.read_bytes()
    raw = json.loads(payload)
    if raw["evidence"] != "MEASURED_ONLINE" or raw["workload"] != "realistic" or raw["gpu_profiler"]:
        raise ValueError(f"Expected a clean real-image cohort: {path}")
    requests = {r["ordinal"]: r for r in raw["requests"]}
    encode = Intervals((r["e_gpu_begin"], r["e_end"]) for r in requests.values() if "e_gpu_begin" in r)
    prefill = Intervals((r["p_start"], r["p_end"]) for r in requests.values() if "p_start" in r)
    steps = {}
    for d in raw["decode_steps"]:
        begin, end = d["begin"], d["end"]
        rows = tuple(d + h for d, h in zip(*d["shards"], strict=True))
        populated = tuple(requests[i] for i in d["ordinals"])
        lengths = tuple(n - r["input_rows"] for n, r in zip(rows, populated, strict=True))
        if len(rows) != d["batch"] or min(lengths) < 1:
            raise ValueError(f"Invalid prepared step shape: {path}")
        shape = tuple((r["ordinal"], r["id"], n, spend) for r, n, spend in zip(populated, rows, lengths, strict=True))
        signature = (d["rank"], shape)
        if signature in steps:
            raise ValueError(f"Duplicate prepared step: {path}")
        e_ms = 1000 * encode.overlap(begin, end)
        p_ms = 1000 * prefill.overlap(begin, end)
        steps[signature] = {
            "batch": d["batch"], "rows": sum(rows), "max_rows": max(rows), "ms": 1000 * (end - begin),
            "assisted": d["assisted"], "backend": d["attention_backend"],
            "stage_at_start": phase_label(encode.contains(begin), prefill.contains(begin)),
            "stage_overlap": phase_label(e_ms, p_ms),
            "e_ms": e_ms, "p_ms": p_ms, "runtime_e_flag": d["encoder_active_at_launch"],
            "prefix": tuple(tuple(r["output_ids"][:n]) for r, n in zip(populated, lengths, strict=True)),
        }
    return raw, requests, steps, hashlib.sha256(payload).hexdigest()


def aggregate(steps):
    groups = defaultdict(list)
    for step in steps.values():
        key = (step["backend"], step["assisted"], step["batch"], context_bin(step["rows"]), step["stage_at_start"])
        groups[key].append(step)
    return [
        {"attention_backend": backend, "assisted": assisted, "actual_batch": batch,
         "total_context_rows_upper_inclusive": bound, "stage_at_start": phase,
         "host_decode_step_ms": summary(row["ms"] for row in group),
         "max_request_context_rows": summary(row["max_rows"] for row in group),
         "E_fraction_of_host_decode_time": sum(row["e_ms"] for row in group) / sum(row["ms"] for row in group),
         "P_fraction_of_host_decode_time": sum(row["p_ms"] for row in group) / sum(row["ms"] for row in group)}
        for (backend, assisted, batch, bound, phase), group in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


def compare(reference, candidate):
    groups = defaultdict(list)
    phase_different = Counter()
    shared_shape = reference.keys() & candidate.keys()
    for key in sorted(shared_shape):
        a, b = reference[key], candidate[key]
        if a["prefix"] != b["prefix"] or a["backend"] != b["backend"]:
            continue
        if a["stage_at_start"] != b["stage_at_start"]:
            phase_different[a["stage_at_start"], b["stage_at_start"]] += 1
        phase = a["stage_at_start"] if a["stage_at_start"] == b["stage_at_start"] else "different"
        groups[(b["assisted"], b["batch"], "all")].append((a, b))
        groups[(b["assisted"], b["batch"], phase)].append((a, b))
        if a["stage_overlap"] == b["stage_overlap"] == "neither":
            groups[(b["assisted"], b["batch"], "neither_entire_step")].append((a, b))
    return {
        "same_ordered_batch_and_context": len(shared_shape),
        "same_shape_backend_and_emitted_prefix": sum(len(rows) for key, rows in groups.items() if key[-1] == "all"),
        "different_stage_at_start": [{"reference": a, "candidate": b, "n": n}
                                     for (a, b), n in sorted(phase_different.items())],
        "paired_groups": [{"actual_batch": batch, "candidate_assisted": assisted, "coactivity_match": phase,
                           "reference_host_decode_step_ms": summary(a["ms"] for a, b in pairs),
                           "candidate_host_decode_step_ms": summary(b["ms"] for a, b in pairs),
                           "paired_delta_ms": summary(b["ms"] - a["ms"] for a, b in pairs),
                           "pair_total_context_rows": summary(a["rows"] for a, b in pairs)}
                          for (assisted, batch, phase), pairs in sorted(groups.items())],
    }


REQUEST_STAGES = {
    "E_queue_ms": ("arrived", "e_start"),
    "render_ms": ("e_start", "render_end"),
    "E_encode_host_ms": ("e_gpu_begin", "e_end"),
    "P_queue_ms": ("e_end", "p_start"),
    "P_prefill_host_ms": ("p_start", "p_end"),
    "D_queue_ms": ("p_end", "d_admission"),
    "initial_place_host_ms": ("d_admission", "d_ready"),
}


def request_comparison(reference, candidate):
    bins = defaultdict(list)
    for key in sorted(reference.keys() & candidate.keys()):
        a, b = reference[key], candidate[key]
        if (a["id"], a["input_rows"], a["image_count"]) != (b["id"], b["input_rows"], b["image_count"]):
            raise ValueError("The late-stage measurements have different image prompts")
        for metric, (start, end) in REQUEST_STAGES.items():
            if all(field in r for field in (start, end) for r in (a, b)):
                row = (1e3 * (a[end] - a[start]), 1e3 * (b[end] - b[start]))
                bins["all", metric].append(row)
                bins[context_bin(a["input_rows"]), metric].append(row)
        for metric in ("ttft_ms", "tpot_ms", "e_gpu_enclosure_ms"):
            if metric in a and metric in b and a[metric] is not None and b[metric] is not None:
                row = (a[metric], b[metric])
                bins["all", metric].append(row)
                bins[context_bin(a["input_rows"]), metric].append(row)
    return [{"input_rows_upper_inclusive": bound, "metric": metric,
             "reference": summary(a for a, b in rows), "candidate": summary(b for a, b in rows),
             "paired_delta": summary(b - a for a, b in rows)}
            for (bound, metric), rows in sorted(bins.items(), key=lambda item: str(item[0]))]


def create(paths):
    loaded = [load(path) for path in paths]
    baseline, reference_requests, baseline_steps, _ = loaded[0]
    fields = ("family", "topology", "offered_qps", "seed", "cap", "vision_backend", "encoder_pipeline", "attention_engine")
    offsets = {r["ordinal"]: round(r["planned"] - baseline["start_monotonic"], 6)
               for r in reference_requests.values()}
    if baseline["helper"] != "none" or len(set(path.parent.name for path in paths)) != len(paths):
        raise ValueError("Put a local EPD baseline first, with distinct arm directories")
    result = {"evidence": "DIAGNOSTIC", "baseline": paths[0].parent.name, "arms": {}}
    for path, (raw, requests, steps, digest) in zip(paths, loaded, strict=True):
        if any(raw.get(k) != baseline.get(k) for k in fields):
            raise ValueError(f"Incompatible serving conditions: {path}")
        if offsets != {r["ordinal"]: round(r["planned"] - raw["start_monotonic"], 6) for r in requests.values()}:
            raise ValueError(f"Different planned arrival offsets: {path}")
        total_ms = sum(row["ms"] for row in steps.values())
        starts = Counter(row["stage_at_start"] for row in steps.values())
        deltas = Counter(row["stage_overlap"] for row in steps.values())
        routes = Counter((row["assisted"], row["backend"], row["runtime_e_flag"]) for row in steps.values())
        entry = {"raw_file": f"{path.parent.name}/{path.name}", "raw_sha256": digest,
                 "complete": raw["complete"], "planned": raw["planned"], "helper": raw["helper"],
                 "partition": raw["partition"], "steps": len(steps), "step_ms": summary(row["ms"] for row in steps.values()),
                 "assisted_requests": sum(r.get("helper_prompt_rows", 0) > 0 for r in requests.values()),
                 "routes": [{"assisted": assisted, "attention_backend": backend,
                             "encoder_active_at_launch": active, "count": count}
                            for (assisted, backend, active), count in sorted(routes.items())],
                 "phase_at_start_counts": dict(starts), "whole_step_coactivity_counts": dict(deltas),
                 "E_overlap_fraction": sum(row["e_ms"] for row in steps.values()) / total_ms,
                 "P_overlap_fraction": sum(row["p_ms"] for row in steps.values()) / total_ms,
                 "E_host_at_start_launch_flag_disagreements": sum(
                     row["stage_at_start"].startswith("E") != row["runtime_e_flag"] for row in steps.values()),
                 "steps_by_batch_context_and_start_phase": aggregate(steps)}
        if path != paths[0]:
            entry["matched_steps"] = compare(baseline_steps, steps)
            entry["matched_requests"] = request_comparison(reference_requests, requests)
        result["arms"][path.parent.name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = create(args.runs)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
