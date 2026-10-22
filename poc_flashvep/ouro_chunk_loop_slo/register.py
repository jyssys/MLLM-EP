"""Freeze SLOs and counterbalanced jobs before new policy comparisons."""

import json
import hashlib
import numpy as np
from .common import ROOT, digest, write


def main():
    destination = ROOT / "slo_registration.json"
    assert not destination.exists(), "Registration is immutable"
    reference = ROOT / "raw/cal_solo_gpu4_v2"
    measurements = [
        json.loads(p.read_text())
        for p in sorted(reference.glob("decode_P2048_N4_r*.json"))
    ]
    assert len(measurements) == 3
    reference_time = float(np.median([r["decode_p50_s"] for r in measurements]))
    prefills = {}
    for length in (2048, 8192):
        x = [
            json.loads(p.read_text())["requests"][0]["TTFT_s"]
            for p in sorted(reference.glob(f"prefill_P{length}_r*.json"))
        ]
        assert len(x) == 3
        prefills[length] = dict(
            reference_TTFT_s=float(np.median(x)),
            thresholds_s=[float(np.median(x)) * f for f in (1.25, 1.5, 2)],
        )
    record = dict(
        reference="GPU4 solo, P2033, resident N4, median of three warmed cohort ITL medians",
        t_dec_ref_s=reference_time,
        relative_TBT_s={
            name: reference_time * f
            for name, f in (("Strict", 5), ("Moderate", 10), ("Relaxed", 25))
        },
        absolute_TBT_s=[0.075, 0.100, 0.125, 0.150, 0.200, 0.250, 0.500],
        TTFT_fixed_s=[2, 5, 10],
        TTFT_calibrated=prefills,
        primary_TTFT_s=[2, 5],
        primary_TBT_tiers=["Strict", "Moderate", "Relaxed"],
        attainment_targets=[0.80, 0.90, 0.95],
        queue_stability=dict(
            max_backlog_slope_requests_s=0.05,
            max_final_unfinished_fraction=0.10,
            maximum_timeouts=0,
            minimum_completed_fraction_by_drain=0.99,
            note="Short finite windows cannot prove long-run stability. Strict capacity needs repeated nondivergent queues and a confidence-aware rate bracket.",
        ),
        capacity_qualification=dict(
            minimum_requests_per_arm_rate=1000,
            preferred_requests=5000,
            confidence="request-family and server-start bootstrap, 5000 draws seed20261023",
            small_sample="EXPLORATORY_RATE_BRACKET_NOT_PRODUCTION_CAPACITY",
        ),
        server=dict(
            backend="flash_attn_4 4.0.0b30",
            max_batch_token_budget=1024,
            KV_global_blocks=18432,
            KV_pool_GiB=54,
            max_active_requests=8,
            precision="BF16",
            recurrent_depth=4,
        ),
        calibration_files={
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted(reference.glob("*.json"))
        },
        historical_result="CONDITIONAL_FIXED_QUANTUM_ONLY_UNCHANGED",
        adaptive="NOT_AUTHORIZED_UNTIL_GATES",
        GPUs="Independent single-GPU workers4–7, not a DP capacity result",
    )
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    write(destination, record)
    modes = (
        [("atomic", c) for c in (128, 256, 512, 1024)]
        + [(m, c) for m in ("split", "grouped", "mixed") for c in (128, 256, 512, 1024)]
        + [("staged4", 1024)]
    )
    rng = np.random.default_rng(20261023)
    jobs = []
    for startup in range(3):
        for shape in ((2048, 4), (8192, 8), (2048, 8), (8192, 4)):
            order = rng.permutation(len(modes))
            jobs.append(
                dict(
                    id=f"grid_v2_s{startup}_P{shape[0]}_N{shape[1]}",
                    kind="grid",
                    shape=list(shape),
                    startup=startup,
                    trials=[(2, 6, 9)[startup]],
                    cells=[list(modes[i]) for i in order],
                    trace=False,
                )
            )
    write(
        ROOT / "fixed_grid_manifest.json",
        dict(seed=20261023, registration_sha256=record["sha256"], jobs=jobs),
    )
    markdown = [
        "# Frozen SLO registration",
        "",
        f"The reference is the median of three warmed GPU4-alone ITL medians at 2,033 prompt tokens and batch width4: {reference_time * 1000:.6f} ms.",
        "This differs from Sarathi's 4K/B32 reference. These are exploratory Ouro thresholds, not product requirements.",
        "",
        "| TBT tier | Multiplier | Limit ms |",
        "| --- | ---: | ---: |",
    ]
    for name, f in (("Strict", 5), ("Moderate", 10), ("Relaxed", 25)):
        markdown.append(f"| {name} | {f} | {reference_time * f * 1000:.6f} |")
    markdown += [
        "",
        "The absolute sensitivity grid is 75/100/125/150/200/250/500 ms.",
        "The fixed TTFT grid is 2/5/10 s. The primary grid uses 2/5 s and all three relative TBT tiers.",
        "The 1.25×/1.5×/2× no-interference TTFT thresholds are also frozen per prompt bin in the JSON.",
        "All offered requests count. Request-max and pooled-p99 contracts remain separate.",
        "An unsafe or diverging queue cannot establish sustainable capacity.",
        "A rate with fewer than 1,000 requests per arm remains exploratory for p99/capacity claims.",
        "No SLO value depends on candidate measurements. The legacy decision remains unchanged.",
        "",
    ]
    (ROOT / "slo_registration.md").write_text("\n".join(markdown))
    print("REGISTERED", record["relative_TBT_s"], len(jobs), "grid process blocks")


if __name__ == "__main__":
    main()
