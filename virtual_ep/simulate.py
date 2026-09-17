"""Command-line entry point for virtual EP2/EP4/EP8 simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .comm_model import CommunicationScenario, SCENARIOS
from .compute_model import ComputeModel
from .event_simulator import EventPolicy
from .schema import TraceBundle
from .simulator import simulate_trace, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ep", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="ep2_calibrated_base")
    parser.add_argument("--communication-model", type=Path)
    parser.add_argument("--compute-model", type=Path)
    parser.add_argument("--event-policy", type=Path)
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = TraceBundle.load(args.trace)
    communication = compute = policy = None
    if not args.structural_only:
        if args.communication_model is None or args.compute_model is None:
            parser.error("timing simulation requires --communication-model and --compute-model")
        # EP1 has no dispatch/combine. A communication model is accepted for a
        # uniform CLI but deliberately not consulted by the simulator.
        communication = (None if args.ep == 1 else
                         CommunicationScenario.load(args.communication_model, args.scenario))
        compute = ComputeModel.from_csv(args.compute_model, allowed_splits=("train",))
        if args.event_policy:
            import json

            policy = EventPolicy(**json.loads(args.event_policy.read_text()))
    predictions = simulate_trace(
        trace,
        args.ep,
        communication=communication,
        compute=compute,
        event_policy=policy,
    )
    write_predictions(
        predictions,
        args.output,
        timing_verdict=("EP2-CALIBRATED" if args.ep == 1 and compute else
                        communication.validation_verdict if communication else None),
    )


if __name__ == "__main__":
    main()
