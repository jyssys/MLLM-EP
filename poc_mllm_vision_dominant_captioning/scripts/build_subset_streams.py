from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import DOMAINS, atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=ROOT / "artifacts/manifests/frozen_image_cohort.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/manifests")
    args = parser.parse_args()
    rows = read_json(args.cohort)["requests"]
    ep2 = []
    for index in list(range(2)) + list(range(24, 26)):
        for domain in DOMAINS:
            ep2.append(next(row["request_id"] for row in rows if row["domain"] == domain and row["domain_index"] == index))
    atomic_json(args.output_dir / "frozen_ep2_representative_stream.json", {
        "schema_version": 1,
        "frozen_before_routing_analysis": True,
        "request_count": len(ep2),
        "request_ids": ep2,
        "composition": "2 discovery + 2 heldout per domain",
    })


if __name__ == "__main__":
    main()
