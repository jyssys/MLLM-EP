"""Reuse verified original prompts and choose causal suffixes by closing delimiter."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ..mllm_epd_modality_aware_idle_execution_discovery.readiness_followup_cases import read_original


def cases(matched, shaped):
    varied = json.loads(matched.read_bytes())["cases"]
    same = json.loads(shaped.read_bytes())["cases"]
    front = {item["case_id"]: item for item in same + varied if item["cut_tokens"] <= 4084}
    results = []
    for item in front.values():
        results.append(item | {"test": "prefix"})
    # Two documents with independently selected sizes. These cases predate timing.
    originals = {item["request_id"]: item["request"] for item in same + varied}
    for document, suffix_count in (("725", 1), ("725", 2), ("174", 1)):
        row = originals[f"yubo2333__MMLongBench-Doc__{document}"]
        ids, positions = read_original(row)
        prefix_images = row["image_count"] - suffix_count
        cut = row["image_visual_spans"][prefix_images - 1][1] + 1
        if ids[cut - 1] != 151653 or not 0 < cut < len(ids):
            raise ValueError("The original Qwen row has no legal closing-delimiter cut")
        results.append({
            "case_id": f"suffix_{document}_{suffix_count}", "test": "suffix", "request": row,
            "request_id": row["request_id"], "cut_tokens": cut, "prefix_images": prefix_images,
            "total_tokens": len(ids), "total_images": row["image_count"],
            "prefix_token_ids_SHA256_int64_le": sha256(ids[:cut].astype("<i8").tobytes()).hexdigest(),
            "prefix_positions_SHA256_int64_le": sha256(positions[..., :cut].astype("<i8").tobytes()).hexdigest(),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--shaped", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--compact-plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = {"cases": cases(args.matched, args.shaped), "selection_scope": "Pre-registered original image-close cuts"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.plan or args.compact_plan:
        if not args.manifest:
            parser.error("A run plan requires the frozen --manifest")
        plan = {
            "cases": [{"tag": "joint", "manifest": str(args.manifest.resolve(strict=True)),
                       "cohort_id": "joint", "count": 6, "qps": 1.465, "output": 16}],
            "joint": {
                "cases": str(args.out.resolve(strict=True)),
                "repetitions": 5,
                "correctness": ["same_doc_first_image", "small_1152"],
                "diagnostics": ["same_doc_first_image", "medium_4084"],
                "interference": [{"fixture": "same_doc_le1024", "native_request": "yubo2333__MMLongBench-Doc__725",
                                  "repetitions": 3}],
            },
        }
        if args.plan:
            args.plan.parent.mkdir(parents=True, exist_ok=True)
            args.plan.write_text(json.dumps(plan, indent=2) + "\n")
        if args.compact_plan:
            compact = {"cases": plan["cases"], "joint": {
                "cases": plan["joint"]["cases"], "repetitions": 5,
                "only": ["suffix_725_1", "suffix_725_2", "suffix_174_1"],
                "include_compact_suffix": True, "suffix_correctness": ["suffix_725_1"],
            }}
            args.compact_plan.parent.mkdir(parents=True, exist_ok=True)
            args.compact_plan.write_text(json.dumps(compact, indent=2) + "\n")


if __name__ == "__main__":
    main()
