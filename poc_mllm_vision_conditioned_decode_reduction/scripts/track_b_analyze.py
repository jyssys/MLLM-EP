#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_vision_conditioned_decode_reduction.track_b import run_track_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded Track-B recurrence/oracle analysis")
    parser.add_argument("--trace-dir", action="append", default=[], type=Path)
    parser.add_argument("--compact-routes", type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--vector-capture", type=Path)
    args = parser.parse_args()
    if not args.trace_dir and args.compact_routes is None:
        parser.error("provide --compact-routes or at least one --trace-dir")
    result = run_track_b(
        trace_dirs=args.trace_dir,
        pool_path=args.pool,
        output_root=args.output_root,
        vector_capture=args.vector_capture,
        compact_routes=args.compact_routes,
    )
    print(result)


if __name__ == "__main__":
    main()
