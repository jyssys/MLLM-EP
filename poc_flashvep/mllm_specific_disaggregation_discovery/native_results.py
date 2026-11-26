"""Reconcile client and EPD timestamps and score the frozen Native cohort."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .quality import Scorer
from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt


def jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def distribution(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": float(np.mean(values)), "p50": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def delta(row, begin, end):
    if row.get(begin) is not None and row.get(end) is not None:
        return (row[end] - row[begin]) * 1000
    return None


def block_count(value):
    if isinstance(value, list):
        if all(isinstance(index, int) for index in value):
            return len(value)
        return sum(block_count(elem) for elem in value)
    raise ValueError(f"Unexpected block list encoding {type(value)}")


def collect(run, label, manifest, scorer):
    source = run / "raw" / f"{label}_requests.jsonl"
    done = rt.read(run / "state" / f"{label}_complete.json")
    config = rt.read(run / "configs" / f"{label}.json")
    arrival = run / "configs" / f"{label}_arrival.json"
    results = jsonl(source)
    rt.require(rt.sha(source) == done["raw_SHA256"] and rt.sha(arrival) == config["arrival_SHA256"],
               f"{label} raw client or arrival receipt changed")
    metadata = {row["request_id"]: row for row in manifest["requests"]}
    states = {row["runtime_request_id"]: row for row in jsonl(run / "raw/proxy_epd.jsonl")
              if row["runtime_request_id"].startswith(f"{label}_")}
    out = []
    for result in results:
        row = metadata[result["request_id"]]
        stage = states.get(result["runtime_request_id"], {})
        completed = result["status"] == "MEASURED_ONLINE"
        if completed:
            rt.require(row["prompt_SHA256"] == result["prompt_SHA256"] and
                       row["total_input_tokens"] == result["usage"]["prompt_tokens"],
                       "Observed prompt identity or geometry changed")
        if completed and scorer is not None:
            score = scorer(row, result["text"])
        else:
            score = {"score": None}
        blocks = stage.get("P_response", {}).get("kv_transfer_params", {}).get("remote_block_ids")
        out.append({
            "request_id": row["request_id"], "source": row["source"], "study_source": row["study_source"],
            "study_split": row["study_split"], "image_count": row["image_count"], "status": result["status"],
            "input_tokens": row["total_input_tokens"], "visual_tokens": row["visual_tokens"],
            "text_tokens": row["text_tokens"], "special_tokens": row["special_tokens"],
            "output_tokens": result.get("output_tokens"), "cap_hit": result.get("output_cap_hit"),
            "TTFT_ms": result.get("TTFT_ms"), "ITL_ms": result.get("ITL_ms"), "TPOT_ms": result.get("TPOT_ms"),
            "E2E_ms": result.get("E2E_ms"), "answer": result["text"], "score": score,
            "E_HTTP_ms": delta(stage, "E_submit", "E_complete"),
            "E_to_P_ms": delta(stage, "E_complete", "P_submit"),
            "P_HTTP_ms": delta(stage, "P_submit", "P_ready"),
            "D_admission_ms": delta(stage, "P_ready", "D_submit"),
            "D_first_SSE_ms": delta(stage, "D_submit", "D_first_SSE_bytes"),
            "D_HTTP_ms": delta(stage, "D_submit", "D_http_complete"),
            "exported_prompt_block_ids": block_count(blocks) if blocks is not None else None,
            "client_scheduled_arrival": result["scheduled_arrival"], "client_finish": result["finish"],
            "worker_proxy_times": {key: stage.get(key) for key in (
                "E_submit", "E_complete", "P_submit", "P_ready", "D_submit", "D_first_SSE_bytes",
                "D_http_complete", "finished")},
        })
    nvml = rt.read(run / "raw" / f"{label}_nvml.json")
    samples = {role: [row["roles"][role]["memory_used_bytes"] for row in nvml["rows"]]
               for role in ("encoder", "prefill", "decode")}
    return out, {
        "QPS": config["QPS"], "client_raw_SHA256": done["raw_SHA256"], "arrival_SHA256": rt.sha(arrival),
        "finite_cohort": done, "metrics": rt.read(run / "analysis" / f"{label}_metrics.json"),
        "peak_sampled_device_memory_bytes": {role: max(values, default=None) for role, values in samples.items()},
        "NVML_sample_count": len(nvml["rows"]), "NVML_errors": nvml["errors"],
        "output_cap_hits": sum(bool(row["cap_hit"]) for row in out),
    }


def summarize(rows, fields):
    return {
        field: distribution([row[field] for row in rows if row.get(field) is not None])
        for field in fields
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--scorers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="discovery200_native")
    args = parser.parse_args()
    scorer = Scorer(args.asset_repo, args.scorers)
    manifest = rt.read(args.manifest)
    rows, summary = collect(args.run, args.label, manifest, scorer)
    rt.require(len(rows) == 200 and len({row["request_id"] for row in rows}) == 200,
               "Native results require 200 unique completed or failed request rows")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    summary["confirmation_manifest_SHA256"] = rt.sha(args.manifest)
    summary["scorer_provenance"] = scorer.provenance
    summary["by_source"] = {}
    for source, group in sorted(groups.items()):
        scores = [row["score"]["score"] for row in group if row["score"]["score"] is not None]
        summary["by_source"][source] = {
            "requests": len(group), "n_scored": len(scores),
            "score": float(np.mean(scores)) if scores else None,
            "metric": group[0]["score"].get("metric"),
            "complete_task_scorer": all(row["score"].get("complete_task_scorer", False) for row in group),
            **summarize(group, ("input_tokens", "visual_tokens", "text_tokens", "special_tokens",
                                "output_tokens", "image_count", "TTFT_ms", "E2E_ms")),
        }
    summary.update(summarize(rows, ("input_tokens", "visual_tokens", "text_tokens", "special_tokens",
                                    "output_tokens", "image_count", "E_HTTP_ms", "E_to_P_ms", "P_HTTP_ms",
                                    "D_admission_ms", "D_first_SSE_ms", "D_HTTP_ms", "E2E_ms", "TPOT_ms")))
    summary["stage_scope"] = "Coordinator timestamps for HTTP intervals. Device-wide NVML memory is sampled."
    rt.write(args.out / "native_summary.json", summary)
    rt.write(args.out / "native_requests.json", rows)


if __name__ == "__main__":
    main()
