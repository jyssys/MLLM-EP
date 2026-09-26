from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.evaluation.public_benchmarks import (
    build_chartqa_manifest,
    build_mmstar_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmstar-parquet", type=Path, required=True)
    parser.add_argument("--chartqa-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mmstar = build_mmstar_manifest(
        args.mmstar_parquet,
        args.output_dir / "mmstar_full_manifest.json",
        args.output_dir / "mmstar_images",
    )
    chartqa = build_chartqa_manifest(
        args.chartqa_source,
        args.output_dir / "chartqa_full_manifest.json",
    )
    print(json.dumps({"mmstar": len(mmstar["requests"]), "chartqa": len(chartqa["requests"])}, indent=2))


if __name__ == "__main__":
    main()

