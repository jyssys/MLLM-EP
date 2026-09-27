from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_mixed_domain_online_calibration.trace_data import consolidate, save_compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("ep2", "ep4"), required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    rows, audit = consolidate(args.trace_dir, args.results, args.topology)
    save_compact(rows, audit, args.output, args.audit)


if __name__ == "__main__":
    main()
