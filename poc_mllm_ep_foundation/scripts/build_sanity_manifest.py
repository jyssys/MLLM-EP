from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_ep_foundation.evaluation.workloads import build_sanity_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_sanity_manifest(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

