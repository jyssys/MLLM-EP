#!/usr/bin/env python3
"""Generate the exact EP4/top-8 fanout-control family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_rankfanout.rankfanout.routing import (  # noqa: E402
    balanced_fanout_route,
    linear_expert_to_rank,
    summarize_route,
    validate_control_family,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, object]] = []
    mapping = linear_expert_to_rank()
    for num_tokens in args.tokens:
        routes = [balanced_fanout_route(num_tokens, f, seed=args.seed + num_tokens) for f in range(1, 5)]
        control = validate_control_family(routes)
        for fanout, route in enumerate(routes, 1):
            key = f"m{num_tokens}_f{fanout}"
            arrays[key] = route.astype(np.int16)
            manifest.append({"key": key, "requested_fanout": fanout, **summarize_route(route, mapping).to_dict()})
        manifest.append({"key": f"m{num_tokens}_control", **control})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "arrays": len(arrays), "controls": len(args.tokens)}, indent=2))


if __name__ == "__main__":
    main()
