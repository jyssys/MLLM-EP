"""Compare independent HF and native logits at identical causal positions."""
import argparse
import csv
import json
import os
from pathlib import Path

import torch

from analyze_numerical_screen import compare


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root/"manifest.json").read_text())
    rows, summaries = [], []
    for run in manifest["runs"]:
        screen = Path(run["screen"])
        source_manifest = json.loads((screen/"manifest.json").read_text())
        variants = [r["name"] for r in source_manifest["records"] if r.get("returncode") == 0]
        for variant in variants:
            name = screen.name + ":" + variant
            values = compare(Path(run["artifact"]), screen/(variant+"_logits"), name)
            rows.extend(values)
            count = len(values)
            disagreements = [r for r in values if not r["argmax_match"]]
            summaries.append({"screen": screen.name, "variant": variant,
                              "positions": count,
                              "argmax_agreement": sum(r["argmax_match"] for r in values)/count,
                              "mean_kl": sum(r["kl"] for r in values)/count,
                              "max_abs_logit_error": max(r["max_logit_abs_error"] for r in values),
                              "disagreements": disagreements})
    with (args.root/"hf_vs_native_logits.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"scope": "FP16_SAME_PREFIX_NUMERICAL_SANITY_NOT_TASK_QUALITY",
              "reference_attention": "HF_SDPA_full_causal_prefix",
              "native_attention": "FlashInfer_incremental_KV",
              "comparisons": summaries,
              "warning": "Near-tie argmax disagreement alone is not semantic or kernel failure; no benchmark quality claim."}
    (args.root/"hf_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
