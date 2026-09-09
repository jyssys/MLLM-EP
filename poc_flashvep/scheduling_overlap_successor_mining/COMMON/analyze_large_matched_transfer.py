"""Clean real-image/text volume control; not a native scheduler counterfactual."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics

from analyze_transfer_cpu import save_csv


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    inputs = {r["request_id"]: r for r in json.loads(args.inputs.read_text())["requests"]}
    pairs = defaultdict(dict)
    request_rows = []
    for path in sorted(args.screen.glob("b*_clean/requests_dp*.jsonl")):
        block = path.parent.name.split("_")[0]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["warmup"]:
                continue
            rid = row["request_id"].rsplit(":", 1)[-1]
            meta = inputs[rid]
            repeat, _, batch = row["label"].split(":")
            key = block, repeat, batch, row["dp_rank"], meta["matched_pair"]
            role = "vision" if row["image_count"] else "text"
            assert role not in pairs[key], "Duplicate logical request"
            record = {"block": block, "repeat": repeat, "batch": batch,
                      "dp_rank": row["dp_rank"], "matched_pair": meta["matched_pair"],
                      "role": role, "family": row["family"], "request_id": rid,
                      "actual_tokens": row["prompt_tokens"],
                      "expected_tokens": meta["processor_prompt_tokens"],
                      "vision_tokens": meta["vision_tokens"],
                      **{k: row[k] for k in ("e2e_s", "ttft_s", "tpot_s", "processor_s", "engine_e2e_s")}}
            request_rows.append(record)
            pairs[key][role] = record
    paired = []
    for key, pair in pairs.items():
        assert set(pair) == {"vision", "text"}, key
        a, b = pair["vision"], pair["text"]
        paired.append({"block": key[0], "repeat": key[1], "batch": key[2],
                       "dp_rank": key[3], "matched_pair": key[4],
                       "vision_actual_tokens": a["actual_tokens"],
                       "text_actual_tokens": b["actual_tokens"],
                       "tokens_exact": a["actual_tokens"] == b["actual_tokens"],
                       "vision_to_text_e2e_ratio": a["e2e_s"]/b["e2e_s"],
                       "vision_to_text_engine_e2e_ratio": a["engine_e2e_s"]/b["engine_e2e_s"],
                       "vision_to_text_tpot_ratio": a["tpot_s"]/b["tpot_s"],
                       "scope": "composition_cost_control_NOT_treatment_or_successor_gain"})
    aggregate = []
    for block, family, batch in sorted({(r["block"], r["family"], r["batch"]) for r in request_rows}):
        group = [r for r in request_rows if (r["block"],r["family"],r["batch"]) == (block,family,batch)]
        aggregate.append({"block": block, "family": family, "batch": batch,
                          "requests": len(group),
                          "mean_e2e_s": statistics.mean(r["e2e_s"] for r in group),
                          "zero_entire_ttft_fixed_timeline_bound_pct":
                              100*sum(r["ttft_s"] for r in group)/sum(r["e2e_s"] for r in group),
                          "processor_share_pct":
                              100*sum(r["processor_s"] for r in group)/sum(r["e2e_s"] for r in group)})
    save_csv(args.screen/"matched_clean_requests.csv", request_rows)
    save_csv(args.screen/"matched_composition_pairs.csv", paired)
    save_csv(args.screen/"matched_clean_bounds.csv", aggregate)
    summary = {"clean_requests": len(request_rows), "matched_pairs": len(paired),
               "exact_token_pairs": sum(r["tokens_exact"] for r in paired),
               "processor_prediction_mismatches": sum(r["actual_tokens"] != r["expected_tokens"] for r in request_rows),
               "scope": "vLLM actual request timeline; not native Layered/FastPP/NanoFlow",
               "limitations": ["TTFT zeroing is not a feasible oracle or online queue replay",
                               "includes unavoidable image encoding/preprocessing and normal prefill",
                               "different task content remains; no modality causal quality claim",
                               "same token volume does not mean same expert histogram"]}
    (args.screen/"matched_composition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
