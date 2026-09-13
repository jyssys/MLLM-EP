#!/usr/bin/env python3
"""Enumerate every ordered B in {16,32,64,128} composition of 128 tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compositions(total: int, widths: tuple[int, ...]) -> list[list[int]]:
    output: list[list[int]] = []

    def visit(remaining: int, prefix: list[int]) -> None:
        if remaining == 0:
            output.append(prefix.copy())
            return
        for width in widths:
            if width <= remaining:
                prefix.append(width)
                visit(remaining - width, prefix)
                prefix.pop()

    visit(total, [])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    schedules = compositions(128, (16, 32, 64, 128))
    policies = [
        {
            "name": "_".join(str(width) for width in schedule),
            "block_schedule": schedule,
        }
        for schedule in schedules
    ]
    args.output.write_text(json.dumps(policies, indent=2) + "\n")
    print(f"wrote {len(policies)} exact compositions")


if __name__ == "__main__":
    main()
