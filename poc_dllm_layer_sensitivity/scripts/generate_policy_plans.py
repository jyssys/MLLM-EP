#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def policy(name, mode="none", layers="", phases="early,middle,late"):
    return {
        "name": name,
        "mode": mode,
        "layers": layers,
        "phases": phases,
        "waves": "",
        "capture_phases": phases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=["baseline", "routed_bypass", "stale_routed", "layer_bypass", "routed_only"],
        required=True,
    )
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--phases", default="early,middle,late")
    args = parser.parse_args()

    if "-" in args.layers:
        lo, hi = (int(value) for value in args.layers.split("-", 1))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(value) for value in args.layers.split(",") if value]
    phases = [value for value in args.phases.split(",") if value]

    plans = [policy("baseline")]
    if args.mode != "baseline":
        for phase in phases:
            for layer in layers:
                plans.append(
                    policy(
                        f"{args.mode}_l{layer:02d}_{phase}",
                        args.mode,
                        str(layer),
                        phase,
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plans, indent=2) + "\n")


if __name__ == "__main__":
    main()
