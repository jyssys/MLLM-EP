"""GPU crossover, light-observer, and bounded-output controls."""

import json
from collections import defaultdict

import numpy as np

from .common import ROOT, write
from .history import table


def main():
    calibration = []
    solo = []
    for path in (ROOT / "raw").glob("cal*_v2/decode_P*_N*_r*.json"):
        d = json.loads(path.read_text())
        identity = json.loads((path.parent / "identity.json").read_text())
        calibration.append(
            dict(
                job=path.parent.name,
                GPU=identity["physical_GPU"],
                P=d["prompt_bin"],
                N=d["width"],
                repetition=d["repetition"],
                median_ITL_ms=1000 * d["decode_p50_s"],
                mean_ITL_ms=1000 * d["decode_mean_s"],
                p95_ITL_ms=1000 * d["decode_p95_s"],
                lane="SOLO"
                if "solo" in path.parent.name
                else "FOUR_WORKERS_CONCURRENT",
                evidence="LIVE_MEASURED_NO_INTERFERING_PREFILL_CALIBRATION",
            )
        )
    for r in calibration:
        if r["lane"] == "SOLO" and r["P"] == 2048 and r["N"] == 4:
            solo.append(r["median_ITL_ms"])
    reference = float(np.median(solo))
    for r in calibration:
        r["ratio_to_solo_P2K_N4_reference"] = (
            r["median_ITL_ms"] / reference if r["P"] == 2048 and r["N"] == 4 else None
        )
    table("GPU_interference_calibration.csv", calibration)
    allocation = []
    graph_records = []
    graph_groups = defaultdict(list)
    counts = defaultdict(int)
    grid = defaultdict(dict)
    for path in (ROOT / "raw").glob("grid*/*_T*.json"):
        d = json.loads(path.read_text())
        graph_groups[(path.parent.name, d["mode"], d["chunk_tokens"])].append(d)
        for r in d["requests"]:
            grid[(d["prompt_bin"], d["concurrency"], d["job_id"], d["trial"], r["id"])][
                f"{d['mode']}_C{d['chunk_tokens']}"
            ] = r
        counts[
            (d["GPU"], d["mode"], d["chunk_tokens"], d["prompt_bin"], d["concurrency"])
        ] += 1
    for (gpu, mode, chunk, length, width), n in sorted(counts.items()):
        allocation.append(
            dict(
                GPU=gpu,
                mode=mode,
                C=chunk,
                P=length,
                N=width,
                measured_cohorts=n,
                scope="Policy and GPU crossover, not a DP throughput experiment",
            )
        )
    table("GPU_policy_crossover.csv", allocation)
    for key, records in sorted(graph_groups.items()):
        previous = None
        for record in sorted(records, key=lambda r: r["trial"]):
            stats = record.get("graph_stats", {})
            row = dict(
                job=key[0],
                mode=key[1],
                C=key[2],
                GPU=record["GPU"],
                P=record["prompt_bin"],
                N=record["concurrency"],
                trial=record["trial"],
                scope="Cumulative engine counters include warm-up. Deltas cover only later cohorts in the same reused engine.",
            )
            for field in ("captures", "replays", "fallbacks"):
                row[field + "_cumulative"] = stats.get(field)
                row[field + "_since_previous_recorded_cohort"] = (
                    stats[field] - previous[field] if previous is not None else None
                )
            graph_records.append(row)
            previous = stats
    table("graph_activity.csv", graph_records)
    gpu_effects = defaultdict(list)
    surface_path = ROOT / "cg_response_surface.csv"
    if surface_path.exists():
        import csv

        surface = list(csv.DictReader(surface_path.open()))
        paired_surface = defaultdict(dict)
        for row in surface:
            paired_surface[(row["job"], row["trial"], row["P"], row["N"])][
                f"{row['mode']}_C{row['C']}"
            ] = row
        for records in paired_surface.values():
            baseline = records.get("atomic_C1024")
            if baseline is None:
                continue
            for name, record in records.items():
                gpu_effects[(record["GPU"], record["P"], record["N"], name)].append(
                    dict(
                        p99_ratio=float(record["ITL_p99_ms"])
                        / float(baseline["ITL_p99_ms"]),
                        TTFT_ratio=float(record["long_TTFT_ms"])
                        / float(baseline["long_TTFT_ms"]),
                        cohort_ratio=float(record["drained_cohort_s"])
                        / float(baseline["drained_cohort_s"]),
                    )
                )
    table(
        "GPU_policy_effects.csv",
        [
            dict(
                GPU=key[0],
                P=key[1],
                N=key[2],
                policy=key[3],
                matched_cohorts=len(values),
                mean_paired_p99_ratio=float(np.mean([r["p99_ratio"] for r in values])),
                mean_paired_TTFT_ratio=float(
                    np.mean([r["TTFT_ratio"] for r in values])
                ),
                mean_paired_drained_time_ratio=float(
                    np.mean([r["cohort_ratio"] for r in values])
                ),
                baseline="Same-GPU, same process block/trial Atomic C1024",
                scope="Descriptive device-block effects. Not cross-device causal attribution or independent-token inference.",
            )
            for key, values in sorted(gpu_effects.items())
        ],
    )
    matched = defaultdict(
        lambda: dict(pairs=0, token_exact=0, depth_exact=0, equal_length=0)
    )
    for key, records in grid.items():
        ref = records.get("atomic_C1024")
        if ref is None:
            continue
        for policy, r in records.items():
            if policy == "atomic_C1024":
                continue
            v = matched[(key[0], key[1], policy)]
            v["pairs"] += 1
            v["token_exact"] += r["output_ids"] == ref["output_ids"]
            v["depth_exact"] += r["depths"] == ref["depths"]
            v["equal_length"] += len(r["output_ids"]) == len(ref["output_ids"])
    table(
        "bounded_output_parity.csv",
        [
            dict(
                P=k[0],
                N=k[1],
                candidate=k[2],
                baseline="atomic_C1024",
                **v,
                scope="Bounded128 output trace, not benchmark accuracy",
            )
            for k, v in sorted(matched.items())
        ],
    )
    solo_policy = []
    for path in (ROOT / "raw/solo_policy_verification_gpu4").glob("*_T*.json"):
        concurrent = ROOT / "raw/grid_repeat_s5_P8192_N8" / path.name
        if not concurrent.exists():
            continue
        a = json.loads(path.read_text())
        b = json.loads(concurrent.read_text())
        assert a["GPU"] == b["GPU"] == 4
        ar = {r["id"]: r for r in a["requests"]}
        br = {r["id"]: r for r in b["requests"]}
        assert set(ar) == set(br)
        aitls = [
            v for r in ar.values() if r["background"] for v in r["measured_ITLs_s"]
        ]
        bitls = [
            v for r in br.values() if r["background"] for v in r["measured_ITLs_s"]
        ]
        al = next(r for r in ar.values() if not r["background"])
        bl = next(r for r in br.values() if not r["background"])
        solo_policy.append(
            dict(
                mode=a["mode"],
                C=a["chunk_tokens"],
                GPU=4,
                trial=a["trial"],
                requests=len(ar),
                solo_long_TTFT_ms=1000 * al["TTFT_s"],
                concurrent_long_TTFT_ms=1000 * bl["TTFT_s"],
                solo_cohort_p95_ITL_ms=1000 * float(np.percentile(aitls, 95)),
                concurrent_cohort_p95_ITL_ms=1000 * float(np.percentile(bitls, 95)),
                solo_cohort_p99_ITL_ms=1000 * float(np.percentile(aitls, 99)),
                concurrent_cohort_p99_ITL_ms=1000 * float(np.percentile(bitls, 99)),
                solo_measured_cohort_s=a["measured_cohort_s"],
                concurrent_measured_cohort_s=b["measured_cohort_s"],
                token_exact=sum(ar[r]["output_ids"] == br[r]["output_ids"] for r in ar),
                scope="Same GPU4/inputs, two cohorts/policy. Descriptive temporal/parallel control, not a universal interference constant.",
            )
        )
    table("solo_vs_parallel_policy_controls.csv", solo_policy)
    observers = []
    for path in (ROOT / "raw").glob("paired_observer_block*/observer_True_T*.json"):
        on = json.loads(path.read_text())
        off = json.loads(path.with_name(path.name.replace("True", "False")).read_text())
        observers.append(
            dict(
                job=path.parent.name,
                GPU=on["GPU"],
                trial=on["trial"],
                ON_measured_cohort_s=on["measured_cohort_s"],
                OFF_measured_cohort_s=off["measured_cohort_s"],
                wall_ratio=on["measured_cohort_s"] / off["measured_cohort_s"],
                token_exact=[r["output_ids"] for r in on["requests"]]
                == [r["output_ids"] for r in off["requests"]],
                depth_exact=[r["depths"] for r in on["requests"]]
                == [r["depths"] for r in off["requests"]],
                scope="Matched GPU, input, warm engine and trial. Alternating observer order.",
            )
        )
    table("paired_observer_controls.csv", observers)
    summary = dict(
        solo_decode_reference_ms=reference,
        calibration_rows=len(calibration),
        matched_observer_cohorts=len(observers),
        observer_token_exact=sum(r["token_exact"] for r in observers),
        observer_depth_exact=sum(r["depth_exact"] for r in observers),
        observer_mean_wall_ratio=float(np.mean([r["wall_ratio"] for r in observers]))
        if observers
        else None,
        observer_mean_ratio_CI95=None,
        attribution="GPU differences and parallel interference remain combined except same-GPU4 solo-vs-concurrent control. No unsupported NUMA/power cause.",
    )
    if observers:
        blocks = defaultdict(list)
        for r in observers:
            blocks[r["job"]].append(r["wall_ratio"])
        values = list(blocks.values())
        rng = np.random.default_rng(20261023)
        draws = []
        for _ in range(5000):
            sample = []
            for i in rng.integers(len(values), size=len(values)):
                a = np.array(values[i])
                sample.extend(a[rng.integers(len(a), size=len(a))])
            draws.append(np.mean(sample))
        summary["observer_mean_ratio_CI95"] = [
            float(v) for v in np.percentile(draws, [2.5, 97.5])
        ]
    write(ROOT / "control_summary.json", summary)
    print("CONTROLS", len(calibration), len(allocation), len(observers))


if __name__ == "__main__":
    main()
