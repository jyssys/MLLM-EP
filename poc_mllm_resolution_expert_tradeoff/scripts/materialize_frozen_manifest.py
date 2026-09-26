from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json


RESOLUTION = {"R0": 1.0, "R05": 0.5, "R025": 0.25}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    policy = json.loads(args.frozen_policy.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    candidates = [{"config": "R0_K8", "reason": "HELDOUT_BASELINE_CONTROL"}]
    candidates.extend(
        candidate for candidate in policy["candidates"]
        if candidate["config"] != "R0_K8"
    )
    for candidate in candidates:
        config = str(candidate["config"])
        resolution, kval = config.split("_K", 1)
        k = int(kval)
        rows = []
        for source in base["requests"]:
            row = dict(source)
            row.update({
                "base_request_id": source["request_id"],
                "request_id": f"{source['request_id']}__{resolution}__K{k}",
                "resolution_label": resolution,
                "resolution_ratio_target": RESOLUTION[resolution],
                "k_visual": k,
                "k_text": 8,
                "k_decode": 8,
                "frozen_policy_reason": candidate["reason"],
            })
            rows.append(row)
        target = args.output_dir / f"heldout192_{config.lower()}.json"
        atomic_json(target, {
            "schema_version": "mllm_resolution_expert_tradeoff.manifest.v1",
            "split": "heldout192_frozen_once",
            "config": config,
            "frozen_policy": str(args.frozen_policy.resolve()),
            "requests": rows,
        })
        outputs.append(str(target.resolve()))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
