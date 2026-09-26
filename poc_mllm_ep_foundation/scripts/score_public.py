from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.evaluation.public_benchmarks import score_public_results
from poc_mllm_ep_foundation.io import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_public_results(args.manifest, args.results)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
