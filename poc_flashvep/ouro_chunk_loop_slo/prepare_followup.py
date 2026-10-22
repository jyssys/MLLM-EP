"""Freeze follow-up jobs before held-out traffic comparisons."""

import json
import random
from .common import ROOT, write, digest


def freeze(name, data):
    path = ROOT / name
    if path.exists():
        assert json.loads(path.read_text()) == data, name
    else:
        write(path, data)


def main():
    cells = [
        ["atomic", 256],
        ["atomic", 1024],
        ["split", 512],
        ["grouped", 512],
        ["grouped", 1024],
        ["mixed", 128],
        ["mixed", 512],
        ["staged4", 1024],
    ]
    jobs = []
    for startup, trials in enumerate(([3, 4, 5], [7, 8], [10, 11]), start=3):
        for length, width in ((2048, 4), (8192, 8)):
            order = list(cells)
            random.Random(5000 + startup + length).shuffle(order)
            jobs.append(
                dict(
                    id=f"grid_repeat_s{startup}_P{length}_N{width}",
                    kind="grid",
                    shape=[length, width],
                    startup=startup,
                    trials=trials,
                    cells=order,
                    trace=False,
                )
            )
    freeze(
        "repeated_shortlist_manifest.json",
        dict(
            purpose="Seven additional paired cohorts. Together with trials 2/6/9: ten per shortlisted shape/policy and six independent starts.",
            selection="Fixed representative menu. No adaptive or optimality claim. Includes tuned-family controls and staged G4.",
            jobs=jobs,
        ),
    )
    jobs = []
    for index in range(4):
        order = [
            [m, c] for m in ("atomic", "split", "grouped", "mixed") for c in (128, 1024)
        ]
        order = order[index * 2 : index * 2 + 2] + [["staged4", 1024]]
        jobs.append(
            dict(
                id=f"profile_quantum_block{index}",
                kind="grid",
                shape=[8192, 8],
                startup=f"heavy_{index}",
                trials=[3],
                cells=order,
                trace=True,
            )
        )
    for index in range(4):
        jobs.append(
            dict(
                id=f"observer_control_s{index}",
                kind="grid",
                shape=[2048, 4],
                startup=f"observer_{index}",
                trials=[2, 6, 9],
                cells=[["atomic", 1024]],
                trace=False,
                observer_off=bool(index % 2),
            )
        )
    jobs.append(
        dict(
            id="traffic_pilot_d_v2",
            kind="traffic",
            pilot=True,
            requests=8,
            cap=512,
            trials=[0],
            cells=[["no_chunk", 16384], ["staged4", 1024]],
        )
    )
    freeze(
        "separate_profiling_manifest.json",
        dict(
            purpose="Separate CUDA-event quantum and observer controls. Never headline heavy wall time.",
            jobs=jobs,
        ),
    )
    freeze(
        "arrival_rate_registration.json",
        dict(
            evidence="LIVE_CALIBRATION_THEN_FROZEN_DESIGN",
            lambda_reference_per_s=0.25,
            source="rate_calibration_manifest.json",
            rule="Highest tested Atomic calibration rate satisfying all preregistered finite-window stability rules.",
            caution="0.5/s had near-flat queued backlog, but 4/39 unfinished exceeded the frozen 10% end-window rule. This alone does not prove divergence.",
            heldout_rates=[0.125, 0.25, 0.5, 1.0],
            rationale="Include low load, reference, near-flat calibration boundary, and measured growing-queue overload. Broad rate brackets are exploratory.",
            duration_s=90,
            drain_s=120,
            starts=4,
            cap=512,
            arrival_clock="Scheduled arrival to actual last token. Independent producer without backpressure.",
            qualification="Expected sample counts are below 1000/arm/rate. No production-capacity or qualified p99 claim.",
            policy_menu=cells,
            workload_sha256=digest(ROOT / "traffic_workload_manifest.json"),
            slo_sha256=digest(ROOT / "slo_registration.json"),
        ),
    )
    jobs = []
    for startup in range(4):
        # Different first rates prevent permanent rate-to-device assignment.
        rates = [0.125, 0.25, 0.5, 1.0]
        rates = rates[startup:] + rates[:startup]
        for rate in rates:
            order = list(cells)
            random.Random(8000 + startup * 100 + int(rate * 1000)).shuffle(order)
            jobs.append(
                dict(
                    id=f"heldout_poisson_s{startup}_r{rate:g}",
                    kind="traffic",
                    split="heldout",
                    startup=startup,
                    rate=rate,
                    duration_s=90,
                    drain_s=120,
                    cap=512,
                    arrival_seed=20262000 + 100 * startup,
                    input_seed=20263000 + 100 * startup,
                    trials=[0],
                    cells=order,
                    pattern="poisson",
                )
            )
    # Interleave start/rate blocks. Every block contains every policy on one GPU.
    jobs = [jobs[s * 4 + r] for r in range(4) for s in range(4)]
    freeze(
        "heldout_poisson_manifest.json",
        dict(
            purpose="Independent single-GPU open-loop natural generation. Common arrivals and inputs within each start/rate block.",
            jobs=jobs,
        ),
    )
    jobs = []
    for startup in range(3):
        for rate in (0.25, 0.5):
            order = list(cells)
            random.Random(9000 + startup * 100 + int(rate * 1000)).shuffle(order)
            jobs.append(
                dict(
                    id=f"heldout_burst_s{startup}_r{rate:g}",
                    kind="traffic",
                    split="heldout",
                    startup=startup,
                    rate=rate,
                    duration_s=64,
                    drain_s=120,
                    cap=512,
                    arrival_seed=20264000 + 100 * startup,
                    input_seed=20265000 + 100 * startup,
                    trials=[0],
                    cells=order,
                    pattern="burst",
                )
            )
    freeze(
        "heldout_burst_manifest.json",
        dict(
            purpose="Seeded groups of four arrivals. Natural completion. Exploratory sensitivity, not steady capacity.",
            jobs=jobs,
        ),
    )
    print("FOLLOWUP_MANIFESTS_FROZEN")


if __name__ == "__main__":
    main()
