"""Measured C/G, request-level SLO, and GPU-blocked paired analysis."""

import csv
import gzip
import json
from collections import defaultdict

import numpy as np

from .common import ROOT, write
from .history import table
from .metrics import summarize, compliance

SEED = 20261023
REPLICATES = 5000


def q(values, percentile):
    return float(np.percentile(values, percentile)) if len(values) else None


def label(mode, chunk):
    return f"{mode}_C{chunk}"


def mean_interval(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["job"]].append(float(row[metric]))
    blocks = list(grouped.values())
    rng = np.random.default_rng(SEED)
    choices = rng.integers(len(blocks), size=(REPLICATES, len(blocks)))
    sums, counts = np.zeros(REPLICATES), np.zeros(REPLICATES)
    for slot in range(len(blocks)):
        for index, block in enumerate(blocks):
            selected = np.flatnonzero(choices[:, slot] == index)
            draws = rng.integers(len(block), size=(len(selected), len(block)))
            sums[selected] += np.asarray(block)[draws].sum(axis=1)
            counts[selected] += len(block)
    return np.percentile(sums / counts, [2.5, 97.5])


def pairs(rows):
    indexed = defaultdict(dict)
    for r in rows:
        indexed[(r["P"], r["N"], r["job"], r["trial"])][label(r["mode"], r["C"])] = r
    results = []
    rng = np.random.default_rng(SEED)
    methods = sorted({label(r["mode"], r["C"]) for r in rows})
    for shape in sorted({(r["P"], r["N"]) for r in rows}):
        for baseline in (
            "atomic_C1024",
            "atomic_C256",
            "mixed_C128",
            "mixed_C512",
            "mixed_C1024",
            "split_C512",
            "split_C1024",
            "staged4_C1024",
        ):
            for candidate in methods:
                if baseline == candidate:
                    continue
                matched = [
                    (k, d[baseline], d[candidate])
                    for k, d in indexed.items()
                    if k[:2] == shape and baseline in d and candidate in d
                ]
                if not matched:
                    continue
                groups = defaultdict(list)
                for k, a, b in matched:
                    groups[k[2]].append((a, b))
                blocks = list(groups.values())
                for metric in (
                    "long_TTFT_ms",
                    "ITL_p95_ms",
                    "ITL_p99_ms",
                    "drained_cohort_s",
                ):
                    av = np.array([a[metric] for _, a, _ in matched], float)
                    bv = np.array([b[metric] for _, _, b in matched], float)
                    draws = rng.integers(len(blocks), size=(REPLICATES, len(blocks)))
                    ba, bb = np.zeros(REPLICATES), np.zeros(REPLICATES)
                    bn = np.zeros(REPLICATES)
                    for slot in range(len(blocks)):
                        for index, block in enumerate(blocks):
                            selected = np.flatnonzero(draws[:, slot] == index)
                            if not len(selected):
                                continue
                            choices = rng.integers(
                                len(block), size=(len(selected), len(block))
                            )
                            aa = np.array([a[metric] for a, _ in block])
                            ab = np.array([b[metric] for _, b in block])
                            ba[selected] += aa[choices].sum(axis=1)
                            bb[selected] += ab[choices].sum(axis=1)
                            bn[selected] += len(block)
                    boot = 100 * (1 - bb / ba)
                    results.append(
                        dict(
                            P=shape[0],
                            N=shape[1],
                            baseline=baseline,
                            candidate=candidate,
                            metric=metric,
                            cohorts=len(matched),
                            process_blocks=len(blocks),
                            baseline_mean=float(av.mean()),
                            candidate_mean=float(bv.mean()),
                            reduction_percent=float(100 * (1 - bv.mean() / av.mean())),
                            CI95_low=q(boot, 2.5),
                            CI95_high=q(boot, 97.5),
                            statistic="Ratio of cohort means. Hierarchical paired server-block/cohort bootstrap, 5000 draws.",
                            evidence="LIVE_MEASURED_PAIRED_ANALYSIS",
                        )
                    )
    table("paired_comparisons.csv", results)
    return results


def fixed():
    from .characterize import main as characterize

    characterize()
    with (ROOT / "cg_response_surface.csv").open() as f:
        surface = list(csv.DictReader(f))
    for r in surface:
        for key in ("GPU", "trial", "P", "N", "C", "G"):
            r[key] = int(r[key])
        for key in ("long_TTFT_ms", "ITL_p95_ms", "ITL_p99_ms", "drained_cohort_s"):
            r[key] = float(r[key])
    paired = pairs(surface)
    request_rows, events, quanta, aggregates = [], [], [], []
    for path in sorted((ROOT / "raw").glob("grid*/*_T*.json")):
        d = json.loads(path.read_text())
        base = dict(
            job=path.parent.name,
            GPU=d["GPU"],
            startup=d["startup"],
            trial=d["trial"],
            P=d["prompt_bin"],
            N=d["concurrency"],
            policy=label(d["mode"], d["chunk_tokens"]),
            evidence="LIVE_MEASURED_BOUNDED128",
        )
        for r in d["requests"]:
            vals = r["decode_ITLs_s"]
            row = dict(
                base,
                id=r["id"],
                background=r["background"],
                prompt_tokens=r["prompt_tokens"],
                output_tokens=len(r["output_ids"]),
                finish_reason=r["finish_reason"],
                TTFT_s=r["TTFT_s"],
                TPOT_s=r["TPOT_s"],
                E2E_s=r["E2E_s"],
                max_ITL_s=max(vals, default=0),
                ITL_p95_s=q(vals, 95),
                ITL_p99_s=q(vals, 99),
            )
            for ms in (50, 100, 150, 200, 250):
                row[f"ITLs_over_{ms}ms"] = sum(v > ms / 1000 for v in vals)
            request_rows.append(row)
            for index, (token, tick) in enumerate(
                zip(r["output_ids"], r["token_times"])
            ):
                events.append(
                    dict(
                        base,
                        id=r["id"],
                        token_index=index,
                        token_id=token,
                        seconds_from_arrival=tick - r["arrival"],
                        ITL_s=tick - r["token_times"][index - 1] if index else None,
                        background=r["background"],
                    )
                )
        for i, z in enumerate(d["quantums"]):
            for s in z["spans"]:
                quanta.append(
                    dict(
                        base,
                        quantum=i,
                        **s,
                        host_begin=z["host_begin"],
                        host_submit_end=z["host_submit_end"],
                        GPU_duration_ms="NOT_AVAILABLE_CLEAN_LANE",
                    )
                )
    table("itls_by_request.csv", request_rows)
    # Long token event tables compress without loss. Original raw includes absolute clocks.
    for name, rows in (
        ("request_events_fixed.csv.gz", events),
        ("quantum_events_fixed.csv.gz", quanta),
    ):
        with gzip.open(ROOT / name, "wt", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    for shape in sorted({(r["P"], r["N"]) for r in surface}):
        for method in sorted({label(r["mode"], r["C"]) for r in surface}):
            rows = [
                r
                for r in surface
                if (r["P"], r["N"]) == shape and label(r["mode"], r["C"]) == method
            ]
            if not rows:
                continue
            result = dict(
                P=shape[0],
                N=shape[1],
                policy=method,
                cohorts=len(rows),
                independent_process_blocks=len({r["job"] for r in rows}),
                GPUs=sorted({r["GPU"] for r in rows}),
                evidence="LIVE_MEASURED_MEANS_OF_COHORT_STATISTICS",
            )
            for name in (
                "long_TTFT_ms",
                "ITL_p95_ms",
                "ITL_p99_ms",
                "drained_cohort_s",
                "host_quantum_mean_ms",
                "host_quantum_p95_ms",
                "staged_bytes",
                "peak_allocated_bytes",
                "long_prefill_submissions",
                "all_prefill_submissions_including_staged_prelude",
                "separate_prefill_prelude_submissions",
                "actual_prompt_rows_mean",
                "output_tokens",
            ):
                result[name] = float(np.mean([float(r[name]) for r in rows]))
            result["drained_output_tokens_s"] = float(
                np.mean(
                    [
                        float(r["output_tokens"]) / float(r["drained_cohort_s"])
                        for r in rows
                    ]
                )
            )
            for metric in (
                "long_TTFT_ms",
                "ITL_p95_ms",
                "ITL_p99_ms",
                "drained_cohort_s",
            ):
                low, high = mean_interval(rows, metric)
                result[metric + "_mean_CI95_low"] = float(low)
                result[metric + "_mean_CI95_high"] = float(high)
            aggregates.append(result)
    table("fixed_grid_summary.csv", aggregates)
    pareto = []
    for r in aggregates:
        competitors = [s for s in aggregates if (s["P"], s["N"]) == (r["P"], r["N"])]
        metrics = ("long_TTFT_ms", "ITL_p99_ms")
        dominated = any(
            all(s[m] <= r[m] for m in metrics) and any(s[m] < r[m] for m in metrics)
            for s in competitors
        )
        pareto.append(
            dict(
                r,
                metrics="Minimize mean long TTFT and mean background cohort-p99 ITL",
                dominated=dominated,
                inference="DESCRIPTIVE_POINT_FRONTIER_NOT_STATISTICAL_DOMINANCE",
            )
        )
    table("pareto_frontier.csv", pareto)
    return surface, aggregates, paired


def family_bootstrap(runs, thresholds):
    rng = np.random.default_rng(SEED)
    B = len(runs)
    block_weights = rng.multinomial(B, np.repeat(1 / B, B), size=REPLICATES)
    workload = json.loads((ROOT / "traffic_workload_manifest.json").read_text())
    source_family = {
        r["id"]: f"{r['domain']}:{r.get('source_row', r['id'])}"
        for r in workload["cases"]
    }
    # A 2K and 8K excerpt from one article share a statistical source family.
    all_families = sorted(
        {source_family[r["family"]] for data in runs for r in data["requests"]}
    )
    F = len(all_families)
    family_weights = rng.multinomial(F, np.repeat(1 / F, F), size=REPLICATES)
    numer = np.zeros((REPLICATES, len(thresholds), 3))
    denom = np.zeros(REPLICATES)
    for b, data in enumerate(runs):
        families = defaultdict(list)
        for r in data["requests"]:
            families[source_family[r["family"]]].append(r)
        families = [families[name] for name in all_families]
        sums = []
        for members in families:
            values = []
            for ttft, tbt, _ in thresholds:
                outcomes = [compliance(r, ttft, tbt, True) for r in members]
                values.append(
                    [
                        sum(o["TTFT_pass"] for o in outcomes),
                        sum(o["TBT_observed_all_token_pass"] for o in outcomes),
                        sum(o["joint_pass"] for o in outcomes),
                    ]
                )
            sums.append(values)
        numer += (family_weights @ np.asarray(sums).reshape(F, -1)).reshape(
            REPLICATES, len(thresholds), 3
        ) * block_weights[:, b, None, None]
        denom += (
            family_weights @ np.array([len(m) for m in families])
        ) * block_weights[:, b]
    values = np.full_like(numer, np.nan)
    valid = denom > 0
    values[valid] = numer[valid] / denom[valid, None, None]
    return np.nanpercentile(values, [2.5, 97.5], axis=0), values


def traffic():
    reg = json.loads((ROOT / "slo_registration.json").read_text())
    # Numeric values are fixed in registration. No candidate-specific SLO.
    tiers = [("strict", 5), ("moderate", 10), ("relaxed", 25)]
    # The registration key can change only with a new experiment, not a candidate run.
    ref = reg["t_dec_ref_s"]
    thresholds = [(a, ref * b, name) for a in (2.0, 5.0, 10.0) for name, b in tiers]
    thresholds += [
        (a, b / 1000, f"absolute_{b}ms")
        for a in (2.0, 5.0, 10.0)
        for b in (75, 100, 125, 150, 200, 250, 500)
    ]
    rows, reqs, runs, groups = [], [], [], defaultdict(list)
    work_rows = []
    for path in sorted((ROOT / "raw").glob("heldout*/*_T*.json")):
        d = json.loads(path.read_text())
        if "requests" not in d:
            continue
        job = d["job"]
        base = dict(
            job=job["id"],
            startup=job["startup"],
            GPU=d["GPU"],
            policy=label(d["policy"], d["C"]),
            C=d["C"],
            G=d["G"],
            rate=d["rate"],
            pattern=d["pattern"],
            evidence="LIVE_MEASURED_IN_PROCESS_OPEN_LOOP_NATURAL_GENERATION",
        )
        prompt_work, decode_work = defaultdict(int), defaultdict(int)
        for quantum in d["quantums"]:
            for span in quantum["spans"]:
                if span["phase"] in ("prefill", "prefilling_recurrent"):
                    prompt_work[span["id"]] += span["tokens"] * span["group"]
                elif span["phase"] == "recurrent":
                    decode_work[span["id"]] += span["tokens"] * span["group"]
        for request in d["requests"]:
            rid = request["id"]
            passed = prompt_work[rid] == 4 * request["input_tokens"]
            if request["status"] == "COMPLETE":
                assert passed, (job["id"], rid, "Incomplete full-R4 prefill")
            assert all(depth == 4 for depth in request["depths"]), (
                "Static decode must not skip loops"
            )
            work_rows.append(
                dict(
                    base,
                    id=rid,
                    family=request["family"],
                    status=request["status"],
                    input_tokens=request["input_tokens"],
                    output_tokens=len(request["output_ids"]),
                    prompt_token_passes_submitted=prompt_work[rid],
                    prompt_R4_PASS=passed,
                    decode_token_passes_submitted=decode_work[rid],
                    minimum_decode_passes_for_delivered_outputs=4
                    * max(0, len(request["output_ids"]) - 1),
                    finish_reason=request.get("finish_reason"),
                    evidence="ACTUAL_SUBMISSION_WORK_ACCOUNTING_NOT_HBM_TRAFFIC_OR_E2E_TIME",
                )
            )
        for a, b, name in thresholds:
            r = summarize(d, a, b, require_natural_stop=True)
            rows.append(dict(base, SLO=name, **r))
        stats = summarize(d, 5.0, ref * 10, True)
        runs.append(dict(base, **stats))
        for r in d["requests"]:
            reqs.append(dict(base, **r))
        groups[(d["pattern"], d["rate"], base["policy"])].append(d)
    table("slo_runs_raw.csv", rows)
    table("serving_runs.csv", runs)
    table("request_work_accounting.csv", work_rows)
    if reqs:
        with gzip.open(ROOT / "request_events.csv.gz", "wt", newline="") as f:
            fields = list(dict.fromkeys(k for r in reqs for k in r))
            writer = csv.DictWriter(f, fields, lineterminator="\n")
            writer.writeheader()
            for r in reqs:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()
                    }
                )
    aggregate = []
    draws_by_group = {}
    for (pattern, rate, policy), data in sorted(groups.items()):
        ci, draws = family_bootstrap(data, thresholds)
        draws_by_group[(pattern, rate, policy)] = draws
        all_requests = [r for d in data for r in d["requests"]]
        all_itls = [v for r in all_requests for v in r.get("ITLs_s", [])]
        for index, (a, b, name) in enumerate(thresholds):
            run_rows = [summarize(d, a, b, True) for d in data]
            total = sum(v["offered"] for v in run_rows)
            result = dict(
                pattern=pattern,
                rate=rate,
                policy=policy,
                TTFT_limit_s=a,
                TBT_limit_s=b,
                SLO=name,
                process_blocks=len(data),
                GPUs=sorted({d["GPU"] for d in data}),
                offered=total,
                evidence="LIVE_MEASURED_REQUEST_FAMILY_AND_ENGINE_BLOCK_BOOTSTRAP",
                sample_adequacy="EXPLORATORY_RATE_BRACKET_NOT_PRODUCTION_CAPACITY"
                if total < 1000
                else "CHECK_STABILITY_AND_REPLICATION",
                completed=sum(v["completed"] for v in run_rows),
                errors=sum(v["errors"] for v in run_rows),
                timeouts=sum(v["timeouts"] for v in run_rows),
                natural_complete=sum(v["natural_complete"] for v in run_rows),
                stable_blocks=sum(
                    v["stable_under_preregistered_finite_window_rule"] for v in run_rows
                ),
                queue_growing_blocks=sum(
                    v["queue_slope_requests_s"] is not None
                    and v["queue_slope_requests_s"] > 0.05
                    for v in run_rows
                ),
                pooled_ITL_p95_ms=1000 * q(all_itls, 95),
                pooled_ITL_p99_ms=1000 * q(all_itls, 99),
                pooled_p99_TBT_pass=q(all_itls, 99) <= b,
                TTFT_p50_s=q([r["TTFT_s"] for r in all_requests if "TTFT_s" in r], 50),
                TTFT_p95_s=q([r["TTFT_s"] for r in all_requests if "TTFT_s" in r], 95),
                E2E_p50_s=q([r["E2E_s"] for r in all_requests], 50),
                E2E_p95_s=q([r["E2E_s"] for r in all_requests], 95),
                output_tokens=sum(len(r["output_ids"]) for r in all_requests),
                mean_output_tokens=float(
                    np.mean([len(r["output_ids"]) for r in all_requests])
                ),
                capped=sum(r.get("finish_reason") == "length" for r in all_requests),
            )
            for metric, j in (
                ("TTFT_pass", 0),
                ("TBT_all_token_pass", 1),
                ("joint_pass", 2),
            ):
                result[metric] = sum(r[metric] for r in run_rows)
                result[metric + "_fraction"] = result[metric] / total
                result[metric + "_CI95_low"] = float(ci[0, index, j])
                result[metric + "_CI95_high"] = float(ci[1, index, j])
            window = sum(d["arrival_window_s"] for d in data)
            result["realized_offered_rate_per_s"] = total / window
            result["SLO_goodput_offered_window_req_s"] = result["joint_pass"] / window
            result["SLO_goodput_completed_in_window_req_s"] = (
                sum(
                    r["goodput_completed_within_arrival_window"] * r["arrival_window_s"]
                    for r in run_rows
                )
                / window
            )
            result["rate_qualified90"] = bool(
                result["joint_pass_CI95_low"] >= 0.9
                and result["stable_blocks"] == len(data)
                and total >= 1000
            )
            aggregate.append(result)
    table("slo_attainment_by_rate.csv", aggregate)
    indexed = {
        (r["pattern"], r["rate"], r["policy"], r["TTFT_limit_s"], r["TBT_limit_s"]): r
        for r in aggregate
    }
    comparisons = []
    for pattern, rate in sorted({(r["pattern"], r["rate"]) for r in aggregate}):
        for baseline in ("atomic_C1024", "mixed_C512", "mixed_C128", "split_C512"):
            akey = (pattern, rate, baseline)
            if akey not in groups:
                continue
            for candidate in ("grouped_C1024", "grouped_C512", "staged4_C1024"):
                bkey = (pattern, rate, candidate)
                if bkey not in groups:
                    continue
                if {d["job"]["id"] for d in groups[akey]} != {
                    d["job"]["id"] for d in groups[bkey]
                }:
                    continue
                for i, (a, b, name) in enumerate(thresholds):
                    ar = indexed[(pattern, rate, baseline, a, b)]
                    br = indexed[(pattern, rate, candidate, a, b)]
                    delta = 100 * (
                        draws_by_group[bkey][:, i, 2] - draws_by_group[akey][:, i, 2]
                    )
                    low, high = np.nanpercentile(delta, [2.5, 97.5])
                    comparisons.append(
                        dict(
                            pattern=pattern,
                            rate=rate,
                            baseline=baseline,
                            candidate=candidate,
                            TTFT_limit_s=a,
                            TBT_limit_s=b,
                            SLO=name,
                            offered_per_policy=ar["offered"],
                            joint_attainment_delta_pp=100
                            * (br["joint_pass_fraction"] - ar["joint_pass_fraction"]),
                            paired_CI95_low_pp=float(low),
                            paired_CI95_high_pp=float(high),
                            mean_output_length_delta_tokens=br["mean_output_tokens"]
                            - ar["mean_output_tokens"],
                            baseline_stable_blocks=ar["stable_blocks"],
                            candidate_stable_blocks=br["stable_blocks"],
                            evidence="LIVE_MEASURED_PAIRED_ENGINE_SOURCE_FAMILY_BOOTSTRAP",
                            limitation="Natural outputs can differ. Not equal-work scheduling speedup or a capacity claim.",
                        )
                    )
    table("serving_paired_differences.csv", comparisons)
    capacity = []
    for pattern, policy, a, b in sorted(
        {
            (r["pattern"], r["policy"], r["TTFT_limit_s"], r["TBT_limit_s"])
            for r in aggregate
        }
    ):
        options = [
            r
            for r in aggregate
            if (r["pattern"], r["policy"], r["TTFT_limit_s"], r["TBT_limit_s"])
            == (pattern, policy, a, b)
        ]
        for target in (0.8, 0.9, 0.95):
            eligible = [
                r["rate"]
                for r in options
                if r["joint_pass_fraction"] >= target
                and r["stable_blocks"] == r["process_blocks"]
            ]
            qualified = [
                r["rate"]
                for r in options
                if r["joint_pass_CI95_low"] >= target
                and r["stable_blocks"] == r["process_blocks"]
                and r["offered"] >= 1000
            ]
            capacity.append(
                dict(
                    pattern=pattern,
                    policy=policy,
                    TTFT_limit_s=a,
                    TBT_limit_s=b,
                    attainment_target=target,
                    highest_tested_point_passing_rate=max(eligible, default=None),
                    qualified_capacity_rate=max(qualified, default=None),
                    interpretation="EXPLORATORY_FINITE_WINDOW_RATE_SCREEN_NOT_INTERPOLATED_SUSTAINABLE_CAPACITY",
                )
            )
    table("slo_capacity_brackets.csv", capacity)
    calibrated_ttft(groups, reg)
    return runs, aggregate


def calibrated_ttft(groups, registration):
    """Apply frozen prompt-bin TTFT limits to matching article requests only."""
    workload = json.loads((ROOT / "traffic_workload_manifest.json").read_text())
    bins = {row["id"]: row.get("prompt_bin") for row in workload["cases"]}
    rows = []
    for (pattern, rate, policy), data in sorted(groups.items()):
        for prompt_bin in (2048, 8192):
            limits = registration["TTFT_calibrated"][str(prompt_bin)]
            selected = [
                dict(
                    run,
                    requests=[
                        r for r in run["requests"] if bins[r["family"]] == prompt_bin
                    ],
                )
                for run in data
            ]
            selected = [run for run in selected if run["requests"]]
            if not selected:
                continue
            thresholds = [
                (ttft, tbt, tier)
                for ttft in limits["thresholds_s"]
                for tier, tbt in registration["relative_TBT_s"].items()
            ]
            ci, _ = family_bootstrap(selected, thresholds)
            offered = sum(len(run["requests"]) for run in selected)
            for i, (ttft, tbt, tier) in enumerate(thresholds):
                outcomes = [
                    compliance(r, ttft, tbt, True)
                    for run in selected
                    for r in run["requests"]
                ]
                rows.append(
                    dict(
                        pattern=pattern,
                        rate=rate,
                        policy=policy,
                        P=prompt_bin,
                        offered_article_requests=offered,
                        process_blocks=len(selected),
                        reference_TTFT_s=limits["reference_TTFT_s"],
                        TTFT_limit_s=ttft,
                        TTFT_multiplier=ttft / limits["reference_TTFT_s"],
                        TBT_tier=tier,
                        TBT_limit_s=tbt,
                        TTFT_pass=sum(o["TTFT_pass"] for o in outcomes),
                        joint_pass=sum(o["joint_pass"] for o in outcomes),
                        joint_pass_fraction=sum(o["joint_pass"] for o in outcomes)
                        / offered,
                        joint_CI95_low=float(ci[0, i, 2]),
                        joint_CI95_high=float(ci[1, i, 2]),
                        evidence="LIVE_MEASURED_ARTICLE_BIN_SUBSET_SENSITIVITY",
                        limitation="Frozen stress-prompt calibration applied to article length bins. Not a matched article-specific service requirement or qualified capacity. Conversation prompts have no matched calibration bin.",
                    )
                )
    table("calibrated_TTFT_sensitivity.csv", rows)


def probes():
    numerical = []
    for p in (ROOT / "raw").glob("numerical_C*/*_C*.json"):
        d = json.loads(p.read_text())
        v = d.get("parity")
        if v is None:
            continue
        numerical.append(
            dict(
                source=str(p.relative_to(ROOT)),
                P=d["prompt_tokens"],
                C=d["chunk_tokens"],
                mode=d["mode"],
                tokens=v["tokens"],
                depths=v["depths"],
                KV_exact=v["KV_exact"],
                KV_relative_L2=v["KV_max_relative_L2"],
                hidden_relative_L2=v["hidden"]["rel_l2"],
                logits_relative_L2=v["logits"]["rel_l2"],
                PASS=v["tokens"]
                and v["depths"]
                and v["KV_exact"]
                and v["hidden"]["exact"]
                and v["logits"]["exact"],
                scope="Isolated numerical probe. Not batched full-answer quality.",
            )
        )
    table("numerical_probe_summary.csv", numerical)
    heavy, observer, allocation = [], [], []
    for p in (ROOT / "raw").glob("*/identity.json"):
        d = json.loads(p.read_text())
        completed = p.with_name("completed.json").exists()
        failed = p.with_name("failure.json").exists()
        allocation.append(
            dict(
                job=p.parent.name,
                GPU=d["physical_GPU"],
                UUID=d["UUID"],
                pid=d["pid"],
                kind=d["job"]["kind"],
                startup=d["job"].get("startup"),
                start_wall=d["start"],
                status="COMPLETE"
                if completed
                else "PRESERVED_FAILURE"
                if failed
                else "RUNNING_OR_INTERRUPTED",
                cells=d["job"].get("cells"),
                CPU_affinity=d.get("CPU_affinity"),
            )
        )
    for p in (ROOT / "raw").glob("profile_quantum_*/*_T*.json"):
        d = json.loads(p.read_text())
        long_id = next(r["id"] for r in d["requests"] if not r["background"])
        for z in d["gpu_quantums"]:
            if z["operation"] != "quantum_with_staging_and_stream_waits":
                continue
            descriptor = d["quantums"][z["quantum"]]
            spans = descriptor["spans"]
            phases = sorted({s["phase"] for s in spans})
            kind = (
                "PREFILL_CORE"
                if any(s["phase"] in ("prefill", "prefilling_recurrent") for s in spans)
                else "PREFILL_PRELUDE"
                if "prefilling_prelude" in phases
                else "DECODE_OR_CODA"
            )
            heavy.append(
                dict(
                    job=p.parent.name,
                    GPU=d["GPU"],
                    mode=d["mode"],
                    C=d["chunk_tokens"],
                    G=d["loop_group"],
                    P=d["prompt_bin"],
                    N=d["concurrency"],
                    quantum=z["quantum"],
                    kind=kind,
                    phases=phases,
                    contains_long_prefill=any(
                        s["id"] == long_id
                        and s["phase"]
                        in ("prefill", "prefilling_recurrent", "prefilling_prelude")
                        for s in spans
                    ),
                    long_prefill_rows=sum(
                        s["tokens"]
                        for s in spans
                        if s["id"] == long_id
                        and s["phase"]
                        in ("prefill", "prefilling_recurrent", "prefilling_prelude")
                    ),
                    GPU_elapsed_ms=z["gpu_compute_ms"],
                    GPU_start_ms=z["gpu_start_ms"],
                    GPU_end_ms=z["gpu_end_ms"],
                    actual_prefill_rows=sum(
                        s["tokens"]
                        for s in spans
                        if s["phase"] in ("prefill", "prefilling_recurrent")
                    ),
                    actual_decode_rows=sum(
                        s["tokens"] for s in spans if s["phase"] == "recurrent"
                    ),
                    host_submission_ms=1000
                    * (descriptor["host_submit_end"] - descriptor["host_begin"]),
                    evidence="HEAVY_CUDA_EVENT_STREAM_INTERVAL_INCLUDES_STAGING_AND_WAIT_NOT_PURE_COMPUTE",
                )
            )
    table("quantum_events.csv", heavy)
    quantum_table("quantum_response_surface.csv", heavy, "actual_prefill_rows")
    quantum_table(
        "long_quantum_response_surface.csv",
        [r for r in heavy if r["contains_long_prefill"]],
        "long_prefill_rows",
    )
    for p in (ROOT / "raw").glob("observer_control*/*_T*.json"):
        d = json.loads(p.read_text())
        observer.append(
            dict(
                job=p.parent.name,
                GPU=d["GPU"],
                trial=d["trial"],
                observer_enabled_from_actual_records=bool(d["quantums"]),
                raw_record_quanta_field_type=type(d["record_quanta"]).__name__,
                heavy=d["heavy"],
                measured_cohort_s=d["measured_cohort_s"],
                outputs=[r["output_ids"] for r in d["requests"]],
                depths=[r["depths"] for r in d["requests"]],
            )
        )
    table("observer_controls.csv", observer)
    table("gpu_job_allocation.csv", allocation)
    write(
        ROOT / "numerical_summary.json",
        dict(
            comparisons=len(numerical),
            exact_PASS=sum(r["PASS"] for r in numerical),
            max_KV_relative_L2=max(
                [r["KV_relative_L2"] for r in numerical], default=None
            ),
            full_benchmark_accuracy="NOT_RUN",
        ),
    )
    return heavy, observer, allocation


def quantum_table(name, source, row_field):
    groups = defaultdict(list)
    fields = ("P", "N", "mode", "C", "G", "kind")
    for record in source:
        groups[tuple(record[k] for k in fields)].append(record)
    summary = []
    for key, records in sorted(groups.items()):
        values = np.array([r["GPU_elapsed_ms"] for r in records])
        full = [r["GPU_elapsed_ms"] for r in records if r[row_field] == key[3]]
        summary.append(
            dict(
                zip(fields, key),
                submissions=len(records),
                process_jobs=len({r["job"] for r in records}),
                GPUs=sorted({r["GPU"] for r in records}),
                mean_stream_elapsed_ms=float(values.mean()),
                median_stream_elapsed_ms=float(np.median(values)),
                p95_stream_elapsed_ms=q(values, 95),
                full_C_submissions=len(full),
                full_C_mean_stream_elapsed_ms=float(np.mean(full)) if full else None,
                actual_prefill_rows_mean=float(
                    np.mean([r["actual_prefill_rows"] for r in records])
                ),
                actual_decode_rows_mean=float(
                    np.mean([r["actual_decode_rows"] for r in records])
                ),
                configured_C_times_G=key[3] * key[4],
                row_filter=row_field,
                evidence="HEAVY_MEASURED_DESCRIPTIVE_SUBMISSION_SUMMARY",
                limitation="Stream interval includes staging and waits. Submissions are not independent inferential samples. Not a validated online cost predictor.",
            )
        )
    table(name, summary)


def main():
    surface, aggregate, paired = fixed()
    runs, slo = traffic()
    heavy, observer, allocation = probes()
    table(
        "results_raw.csv",
        [dict(lane="FIXED_BOUNDED128", **r) for r in surface]
        + [dict(lane="NATURAL_OPEN_LOOP", **r) for r in runs],
    )
    write(
        ROOT / "analysis_counts.json",
        dict(
            fixed_cohorts=len(surface),
            serving_runs=len(runs),
            SLO_aggregate_rows=len(slo),
            paired_comparisons=len(paired),
            heavy_quantums=len(heavy),
            heavy_prefill_quantums=sum(r["kind"] == "PREFILL_CORE" for r in heavy),
            heavy_prelude_quantums=sum(r["kind"] == "PREFILL_PRELUDE" for r in heavy),
            heavy_decode_or_coda_quantums=sum(
                r["kind"] == "DECODE_OR_CODA" for r in heavy
            ),
            observer_controls=len(observer),
            jobs=len(allocation),
            bootstrap_replicates=REPLICATES,
            seed=SEED,
        ),
    )
    print("ANALYSIS_COMPLETE", len(surface), len(runs), len(slo))


if __name__ == "__main__":
    main()
