#!/usr/bin/env python3
"""Build request-level gates and figures for the exact-overlap PoC."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/exact_overlap_20260913_134622"
PRIOR_DISCOVERY = Path("/home/esjung/MLLM-EP-discovery/poc_dllm_ep_discovery")
PAIR_CSV = ROOT / "PAIRWISE_OVERLAP_MATRIX.csv"
PHASES = ("early", "middle", "late")
SAMPLED_WAVES = {
    "gsm8k": ("0", "30", "60"),
    "humaneval": ("0", "40", "80"),
}
TOTAL_DENOISE_WAVES = {"gsm8k": 65, "humaneval": 85}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_concatenated(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position, rows = 0, []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text):
            row, position = decoder.raw_decode(text, position)
            rows.append(row)
    return rows


def answer_hash(path: Path) -> tuple[str, int]:
    rows = read_concatenated(path)
    canonical = json.dumps(
        [(int(row["id"]), row["answer"]) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest(), len(rows)


def parse_clean() -> tuple[list[dict], dict[str, float]]:
    rows = []
    pattern = re.compile(
        r"clean_(gsm8k|humaneval)_ep4_b32_mini32_r(\d+)_g32\.log"
    )
    metric = re.compile(r"^Forward: (\d+), Time: ([0-9.]+),.*TPS: ([0-9.]+)")
    for path in sorted((RESULT / "logs").glob("clean_*_g32.log")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        found = None
        for line in path.read_text(errors="replace").splitlines():
            if metric.match(line):
                found = metric.match(line)
        if found is None:
            continue
        dataset, restart = match.groups()
        generation = next((RESULT / "clean" / dataset / f"r{restart}").glob("*.jsonl"))
        digest, requests = answer_hash(generation)
        rows.append(
            {
                "dataset": dataset,
                "restart": restart,
                "requests": requests,
                "nfe": int(found.group(1)),
                "wall_seconds": float(found.group(2)),
                "tokens_per_second": float(found.group(3)),
                "answer_sha256": digest,
                "source": str(path),
            }
        )
    medians = {
        dataset: statistics.median(
            float(row["wall_seconds"]) for row in rows if row["dataset"] == dataset
        )
        for dataset in ("gsm8k", "humaneval")
    }
    write_csv(ROOT / "BASELINE_CLEAN.csv", rows)
    return rows, medians


def parse_quality(clean_rows: list[dict]) -> list[dict]:
    references = {}
    for dataset in ("gsm8k", "humaneval"):
        reference = next(row for row in clean_rows if row["dataset"] == dataset)
        references[dataset] = reference["answer_sha256"]
    rows = []
    for path in sorted(RESULT.glob("**/*.jsonl")):
        dataset = "humaneval" if "humaneval" in str(path) else "gsm8k"
        digest, requests = answer_hash(path)
        rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "requests": requests,
                "answer_sha256": digest,
                "matches_clean_reference": digest == references[dataset],
                "semantic_boundary": "diagnostic replay; production output unchanged",
            }
        )
    write_csv(ROOT / "QUALITY_CORRECTNESS.csv", rows)
    return rows


def parse_phase_pairs() -> list[dict]:
    tagged: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in RESULT.glob("phase_pair/*/waves*/r*/phase_pair_rank*.json"):
        payload = json.loads(path.read_text())
        dataset = path.parts[-4]
        restart = path.parts[-2]
        for row in payload["rows"]:
            tagged[(dataset, row["pair"])].append({**row, "restart": restart})
    output = []
    for (dataset, pair), rows in sorted(tagged.items()):
        by_iteration: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in rows:
            by_iteration[(row["restart"], int(row["iteration"]))].append(row)
        critical = []
        for same_iteration in by_iteration.values():
            serial = max(float(row["serial_ms"]) for row in same_iteration)
            concurrent = max(float(row["concurrent_ms"]) for row in same_iteration)
            critical.append((serial, concurrent, serial - concurrent))
        output.append(
            {
                "dataset": dataset,
                "pair": pair,
                "samples": len(critical),
                "rank_critical_serial_median_ms": statistics.median(x[0] for x in critical),
                "rank_critical_concurrent_median_ms": statistics.median(x[1] for x in critical),
                "real_saving_median_ms": statistics.median(x[2] for x in critical),
                "real_saving_p10_ms": np.quantile([x[2] for x in critical], 0.1),
                "real_saving_p90_ms": np.quantile([x[2] for x in critical], 0.9),
                "min_compute_cosine": min(float(row["compute_cosine"]) for row in rows),
                "max_compute_rel_l2": max(float(row["compute_rel_l2"]) for row in rows),
            }
        )
    write_csv(ROOT / "PHASE_PAIR_OVERLAP_MATRIX.csv", output)
    return output


def phase_summary() -> list[dict[str, str]]:
    rows = read_csv(PRIOR_DISCOVERY / "PHASE_MISMATCH_SUMMARY.csv")
    keep = [
        row for row in rows
        if row["dataset"] in ("gsm8k", "humaneval") and row["phase"] in PHASES
    ]
    write_csv(ROOT / "REFINEMENT_PHASE_PROFILES.csv", keep)
    return keep


def build_oracles(
    pair_rows: list[dict[str, str]], medians: dict[str, float], phase_rows: list[dict]
) -> list[dict]:
    lookup = {
        (row["dataset"], row["wave"], row["pair"]): float(row["real_saving_median_ms"])
        for row in pair_rows
    }
    component = read_csv(PRIOR_DISCOVERY / "COMPONENT_E2E_UPPER_BOUNDS.csv")
    shared_ceiling = {
        row["dataset"]: float(row["clean_e2e_upper_bound_percent"])
        for row in component if row["component"] == "shared"
    }
    output = []
    for dataset in ("gsm8k", "humaneval"):
        sampled_three_stage = [
            float(row["real_saving_median_ms"])
            for row in pair_rows
            if row["dataset"] == dataset
            and row["pair"] == "dispatch_next__expert_current__combine_previous"
        ]
        service_upper = (
            statistics.median(sampled_three_stage)
            * 31
            * TOTAL_DENOISE_WAVES[dataset]
            / (medians[dataset] * 1000)
            * 100
        )
        max_service_upper = (
            max(sampled_three_stage)
            * 31
            * TOTAL_DENOISE_WAVES[dataset]
            / (medians[dataset] * 1000)
            * 100
        )
        output.extend(
            [
                {
                    "dataset": dataset,
                    "candidate": "shared_expert_vs_routed_ep",
                    "scope": "same-request exact",
                    "measured_pairwise": "combine||shared positive in some sampled shapes",
                    "perfect_or_credible_e2e_percent": shared_ceiling[dataset],
                    "direct_request_e2e_percent": shared_ceiling[dataset],
                    "service_throughput_upper_percent": "",
                    "max_every_wave_service_upper_percent": "",
                    "boundary": "full shared contribution is the absolute ceiling; current source serializes it",
                    "prior_art_risk": "DIRECT_ADJACENT_SGLANG_SBO_MEGATRON",
                    "decision": "KILL_LT5_AND_PRIOR_ART",
                },
                {
                    "dataset": dataset,
                    "candidate": "remote_dispatch_vs_local_source_expert",
                    "scope": "same-request exact",
                    "measured_pairwise": "negative median saving in all sampled physical shapes",
                    "perfect_or_credible_e2e_percent": 0.0,
                    "direct_request_e2e_percent": 0.0,
                    "service_throughput_upper_percent": "",
                    "max_every_wave_service_upper_percent": "",
                    "boundary": "exact algebraic independence exists but H100 contention exceeds overlap",
                    "prior_art_risk": "generic locality overlap",
                    "decision": "KILL_CONTENTION",
                },
                {
                    "dataset": dataset,
                    "candidate": "communication_vs_attention",
                    "scope": "cross-request exact",
                    "measured_pairwise": "dispatch||attention negative; combine||attention weak/mixed",
                    "perfect_or_credible_e2e_percent": 0.0,
                    "direct_request_e2e_percent": 0.0,
                    "service_throughput_upper_percent": "",
                    "max_every_wave_service_upper_percent": "",
                    "boundary": "independent requests only; no stable complementarity",
                    "prior_art_risk": "TBO/DBO",
                    "decision": "KILL_CONTENTION",
                },
                {
                    "dataset": dataset,
                    "candidate": "complete_wave_dispatch_expert_combine_pipeline",
                    "scope": "independent-wave steady-state service oracle",
                    "measured_pairwise": "positive exact 3-stage diagnostic",
                    "perfect_or_credible_e2e_percent": service_upper,
                    "direct_request_e2e_percent": 0.0,
                    "service_throughput_upper_percent": service_upper,
                    "max_every_wave_service_upper_percent": max_service_upper,
                    "boundary": "sample-state median projected over independent waves; not direct request latency: t+1 depends on t and best-static mini32 has one wave",
                    "prior_art_risk": "HIGH_TBO_COMET_STREAMEP_XSTAGE",
                    "decision": "CHARACTERIZATION_ONLY",
                },
                {
                    "dataset": dataset,
                    "candidate": "phase_aware_pipeline_selection",
                    "scope": "dLLM phase-aware incremental oracle",
                    "measured_pairwise": "cross-state 2-stage efficiency varies, but no complete pipeline beats generic 3-stage",
                    "perfect_or_credible_e2e_percent": 0.0,
                    "direct_request_e2e_percent": 0.0,
                    "service_throughput_upper_percent": 0.0,
                    "max_every_wave_service_upper_percent": 0.0,
                    "boundary": "no demonstrated phase-aware increment over the generic complete-wave pipeline",
                    "prior_art_risk": "generic phase multiplexing",
                    "decision": "KILL_NO_INCREMENT",
                },
                {
                    "dataset": dataset,
                    "candidate": "next_step_input_independent_preparation",
                    "scope": "same-request exact",
                    "measured_pairwise": "source audit: buffers/workspace persistent; routing metadata input-dependent",
                    "perfect_or_credible_e2e_percent": 0.0,
                    "direct_request_e2e_percent": 0.0,
                    "service_throughput_upper_percent": "",
                    "max_every_wave_service_upper_percent": "",
                    "boundary": "only static descriptors are precomputable and already amortized",
                    "prior_art_risk": "runtime preallocation",
                    "decision": "KILL_LT5",
                },
            ]
        )
    write_csv(ROOT / "CANDIDATE_ORACLES.csv", output)
    return output


def savefig(name: str) -> None:
    out = ROOT / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def make_figures(pair_rows: list[dict], phase_rows: list[dict], phase_pairs: list[dict], oracles: list[dict]) -> None:
    # 1: representative layer timeline from observed stage medians.
    components = ["router", "dispatch", "expert", "combine", "shared", "gather"]
    component_rows = read_csv(PRIOR_DISCOVERY / "COMPONENT_E2E_UPPER_BOUNDS.csv")
    values = {
        dataset: [
            float(next(row["summed_critical_rank_cuda_event_ms"] for row in component_rows if row["dataset"] == dataset and row["component"] == component))
            for component in components
        ]
        for dataset in ("gsm8k", "humaneval")
    }
    plt.figure(figsize=(9, 3.5))
    for y, dataset in enumerate(values):
        left = 0.0
        total_invocations = (66 if dataset == "gsm8k" else 86) * 31
        for component, summed in zip(components, values[dataset]):
            width = summed / total_invocations
            plt.barh(y, width, left=left, label=component if y == 0 else None)
            left += width
    plt.yticks([0, 1], ["GSM8K", "HumanEval"])
    plt.xlabel("Observer-heavy critical-rank CUDA-event ms per layer invocation")
    plt.title("Fine-grained MoE timeline (attribution; not additive request wall)")
    plt.legend(ncol=6, fontsize=7)
    savefig("01_fine_grained_layer_timeline.png")

    # 2: dependency DAG as a compact annotated graph.
    plt.figure(figsize=(12, 4))
    nodes = ["router", "dispatch", "owner expert", "combine", "add shared", "TP gather"]
    for index, node in enumerate(nodes):
        plt.text(index, 0, node, ha="center", va="center", bbox={"boxstyle": "round", "fc": "#e8eef7"})
        if index:
            plt.annotate("", (index - 0.25, 0), (index - 0.75, 0), arrowprops={"arrowstyle": "->"})
    plt.text(2.1, 0.8, "shared expert\nindependent after router input", ha="center", bbox={"boxstyle": "round", "fc": "#fff2cc"})
    plt.annotate("", (4, 0.18), (2.1, 0.65), arrowprops={"arrowstyle": "->", "linestyle": "--"})
    plt.xlim(-0.6, 5.6); plt.ylim(-0.6, 1.3); plt.axis("off")
    plt.title("Confirmed current DeepEP dependency DAG")
    savefig("02_dependency_dag.png")

    # 3: qualitative resource profile; counters were privilege-blocked.
    labels = ["dispatch", "expert", "combine", "attention", "shared"]
    axes = ["TensorCore", "SM", "HBM", "NVLink", "startup"]
    qualitative = np.array([
        [0, 1, 2, 3, 3], [3, 3, 2, 0, 1], [0, 1, 2, 3, 3],
        [3, 3, 2, 0, 1], [3, 3, 2, 0, 1],
    ])
    plt.figure(figsize=(7, 4))
    plt.imshow(qualitative, cmap="Blues", vmin=0, vmax=3, aspect="auto")
    plt.xticks(range(len(axes)), axes); plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="source/pairwise-inferred intensity (0–3, not hardware counter)")
    plt.title("Resource profile atlas (NV GPU counters unavailable: ERR_NVGPUCTRPERM)")
    savefig("03_resource_profile_atlas.png")

    # 4: sampled-state pairwise opportunity matrix. A wave ID is deliberately
    # not relabeled as early/middle/late because ready-pool drain and block
    # progress also determine its physical row count.
    pairs = sorted({row["pair"] for row in pair_rows})
    columns = [
        (dataset, wave)
        for dataset in ("gsm8k", "humaneval")
        for wave in SAMPLED_WAVES[dataset]
    ]
    matrix = np.full((len(pairs), len(columns)), np.nan)
    lookup = {(row["dataset"], row["wave"], row["pair"]): float(row["real_saving_median_ms"]) for row in pair_rows}
    for i, pair in enumerate(pairs):
        for j, (dataset, wave) in enumerate(columns):
            matrix[i, j] = lookup.get((dataset, wave, pair), np.nan)
    plt.figure(figsize=(10, 6))
    limit = np.nanmax(np.abs(matrix))
    plt.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    plt.yticks(range(len(pairs)), pairs, fontsize=7)
    plt.xticks(
        range(len(columns)),
        [f"{dataset[:3]}-w{wave}" for dataset, wave in columns],
    )
    plt.colorbar(label="rank-critical real saving (ms)")
    plt.title("Exact pairwise opportunity matrix (sampled physical waves)")
    savefig("04_pairwise_opportunity_matrix.png")

    # 5/6: serial-concurrent and eta for representative 3-stage + failed pairs.
    selected = ["dispatch_next__expert_current__combine_previous", "remote_dispatch__local_expert", "dispatch__attention", "combine__shared"]
    gsm = [row for row in pair_rows if row["dataset"] == "gsm8k" and row["wave"] == "30" and row["pair"] in selected]
    x = np.arange(len(gsm))
    plt.figure(figsize=(10, 4))
    plt.bar(x - .18, [float(r["rank_critical_serial_median_ms"]) for r in gsm], .36, label="serial")
    plt.bar(x + .18, [float(r["rank_critical_concurrent_median_ms"]) for r in gsm], .36, label="concurrent")
    plt.xticks(x, [r["pair"].replace("__", "\n") for r in gsm], fontsize=7)
    plt.ylabel("ms"); plt.legend(); plt.title("Serial vs actual concurrent (GSM8K wave 30)")
    savefig("05_serial_vs_concurrent_latency.png")
    plt.figure(figsize=(10, 4))
    plt.bar(x, [float(r["eta_median"]) for r in gsm], color=["#4c78a8" if float(r["eta_median"]) > 0 else "#e45756" for r in gsm])
    plt.axhline(0, color="black", linewidth=.8)
    plt.xticks(x, [r["pair"].replace("__", "\n") for r in gsm], fontsize=7)
    plt.ylabel("eta"); plt.title("Contention/overlap efficiency")
    savefig("06_contention_matrix.png")

    # 7: request vs service oracle must be visually separated.
    names = ["shared/routed", "local/remote", "comm/attn", "3-stage waves", "phase-aware", "next prep"]
    direct, service = [], []
    for name in ["shared_expert_vs_routed_ep", "remote_dispatch_vs_local_source_expert", "communication_vs_attention", "complete_wave_dispatch_expert_combine_pipeline", "phase_aware_pipeline_selection", "next_step_input_independent_preparation"]:
        matching = [r for r in oracles if r["candidate"] == name]
        direct.append(statistics.median(float(r["direct_request_e2e_percent"]) for r in matching))
        service.append(statistics.median(float(r["service_throughput_upper_percent"] or 0) for r in matching))
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4))
    plt.bar(x - .18, direct, .36, label="direct request E2E")
    plt.bar(x + .18, service, .36, label="independent-wave service upper")
    plt.axhline(5, color="gray", linestyle="--", label="kill gate")
    plt.xticks(x, names, rotation=20); plt.ylabel("percent"); plt.legend()
    plt.title("Candidate oracle ranking: scope is not interchangeable")
    savefig("07_e2e_oracle_ranking.png")

    # 8: best multistage schematic.
    plt.figure(figsize=(10, 3.5))
    for y, label in enumerate(["comm stream", "compute stream"]):
        plt.text(-.1, y + .25, label, ha="right")
    plt.broken_barh([(0, .4), (.4, .25)], (0, .4), facecolors=["#e45756", "#72b7b2"])
    plt.text(.2, .2, "combine i-1", ha="center", va="center", fontsize=8)
    plt.text(.525, .2, "dispatch i+1", ha="center", va="center", fontsize=8)
    plt.broken_barh([(0, .65)], (1, .4), facecolors="#4c78a8")
    plt.text(.325, 1.2, "expert i", ha="center", va="center", fontsize=8)
    plt.xlim(-.2, .9); plt.ylim(-.2, 1.7); plt.yticks([]); plt.xlabel("normalized diagnostic time")
    plt.title("Best exact 3-stage complete-wave pipeline (independent waves only)")
    savefig("08_best_multistage_pipeline.png")

    # 9/10/11/12: phase profiles and transparent proxy boundaries.
    for dataset in ("gsm8k", "humaneval"):
        subset = [row for row in phase_rows if row["dataset"] == dataset]
        expert = np.array([float(next(r["expert_critical_ms_median"] for r in subset if r["phase"] == p)) for p in PHASES])
        dispatch = np.array([float(next(r["dispatch_critical_ms_median"] for r in subset if r["phase"] == p)) for p in PHASES])
        combine = np.array([float(next(r["combine_critical_ms_median"] for r in subset if r["phase"] == p)) for p in PHASES])
        if dataset == "gsm8k":
            plt.figure(figsize=(7, 4))
        plt.plot(PHASES, expert / (dispatch + combine), marker="o", label=dataset)
    plt.ylabel("expert / (dispatch + combine) time")
    plt.title("Compute/communication time ratio vs refinement phase")
    plt.legend(); savefig("09_compute_communication_ratio_vs_timestep.png")

    for metric, filename, title in [
        ("expert", "10_tensorcore_utilization_vs_phase.png", "Expert-time fraction proxy (TensorCore counters unavailable)"),
        ("comm", "11_nvlink_utilization_vs_phase.png", "Dispatch+combine time fraction proxy (NVLink counters unavailable)"),
    ]:
        plt.figure(figsize=(7, 4))
        for dataset in ("gsm8k", "humaneval"):
            subset = [row for row in phase_rows if row["dataset"] == dataset]
            e = np.array([float(next(r["expert_critical_ms_median"] for r in subset if r["phase"] == p)) for p in PHASES])
            d = np.array([float(next(r["dispatch_critical_ms_median"] for r in subset if r["phase"] == p)) for p in PHASES])
            c = np.array([float(next(r["combine_critical_ms_median"] for r in subset if r["phase"] == p)) for p in PHASES])
            y = e / (e + d + c) if metric == "expert" else (d + c) / (e + d + c)
            plt.plot(PHASES, y * 100, marker="o", label=dataset)
        plt.ylabel("time fraction (%)"); plt.title(title); plt.legend()
        savefig(filename)

    plt.figure(figsize=(8, 4))
    width = .35
    for dindex, dataset in enumerate(("gsm8k", "humaneval")):
        subset = [row for row in phase_rows if row["dataset"] == dataset]
        base = np.arange(3) + (dindex - .5) * width
        bottom = np.zeros(3)
        for key, label in [("dispatch_critical_ms_median", "dispatch"), ("expert_critical_ms_median", "expert"), ("combine_critical_ms_median", "combine")]:
            vals = np.array([float(next(r[key] for r in subset if r["phase"] == p)) for p in PHASES])
            plt.bar(base, vals, width, bottom=bottom, label=label if dindex == 0 else None)
            bottom += vals
    plt.xticks(np.arange(3), PHASES); plt.ylabel("critical-rank ms")
    plt.title("Phase-conditioned DeepEP stage breakdown (paired datasets)")
    plt.legend(); savefig("12_phase_conditioned_stage_breakdown.png")

    # 13: measured cross-state replay, or explicit pending/unavailable panel.
    plt.figure(figsize=(8, 5))
    if phase_pairs:
        waves = sorted({int(re.search(r"wave(\d+)", r["pair"]).group(1)) for r in phase_pairs})
        matrix = np.full((len(waves), len(waves)), np.nan)
        for row in phase_pairs:
            found = re.match(r"(?:dispatch|combine)_wave(\d+)__expert_wave(\d+)", row["pair"])
            if not found or not row["pair"].startswith("combine"):
                continue
            i, j = waves.index(int(found.group(1))), waves.index(int(found.group(2)))
            matrix[i, j] = float(row["real_saving_median_ms"])
        plt.imshow(matrix, cmap="RdBu_r", aspect="auto")
        plt.xticks(range(len(waves)), [f"expert w{w}" for w in waves])
        plt.yticks(range(len(waves)), [f"combine w{w}" for w in waves])
        plt.colorbar(label="real saving (ms)")
        plt.title("Cross-state complete-wave concurrency")
    else:
        plt.text(.5, .5, "Cross-state diagnostic not present", ha="center", va="center")
        plt.axis("off")
    savefig("13_phase_pair_concurrency_matrix.png")

    # 14/15: cross-state two-stage variation exists, but never establishes a
    # better complete pipeline than the generic three-stage result.
    plt.figure(figsize=(7, 4))
    same_two = statistics.median(
        float(r["real_saving_median_ms"])
        for r in pair_rows if r["pair"] == "expert_current__combine_previous"
    )
    cross_two = statistics.median(
        float(r["real_saving_median_ms"]) for r in phase_pairs
        if r["pair"].startswith("combine")
    ) if phase_pairs else 0.0
    cross_best = statistics.median(
        max(
            float(r["real_saving_median_ms"]) for r in phase_pairs
            if r["dataset"] == dataset and r["pair"].startswith("combine")
        )
        for dataset in ("gsm8k", "humaneval")
    ) if phase_pairs else 0.0
    generic_three = statistics.median(
        float(r["real_saving_median_ms"])
        for r in pair_rows
        if r["pair"] == "dispatch_next__expert_current__combine_previous"
    )
    labels = ["same-state\n2-stage", "cross-state\nmedian", "cross-state\nbest", "generic\n3-stage"]
    vals = [same_two, cross_two, cross_best, generic_three]
    plt.bar(labels, vals); plt.ylabel("rank-critical real saving (ms)")
    plt.title("Cross-state pairing changes efficiency but does not beat generic 3-stage")
    savefig("14_blind_vs_phase_aware_pairing.png")
    plt.figure(figsize=(7, 4))
    service = statistics.median(
        float(r["service_throughput_upper_percent"] or 0)
        for r in oracles
        if r["candidate"] == "complete_wave_dispatch_expert_combine_pipeline"
    )
    plt.bar(["generic pipeline\nservice upper", "demonstrated phase-aware\nincrement"], [service, 0])
    plt.ylabel("percent")
    plt.title("No phase-aware complete-pipeline increment established")
    savefig("15_best_static_vs_phase_aware_pipeline.png")

    plt.figure(figsize=(7, 4))
    plt.bar(["persistent workspace", "route-dependent prep", "pre-postable recurring work"], [0, 1, 0])
    plt.ylabel("classification (1 = input dependent)")
    plt.title("Next-step preparation: recurring route layout cannot be prepared exactly")
    savefig("16_next_step_preparation_oracle.png")


def main() -> None:
    clean_rows, medians = parse_clean()
    quality = parse_quality(clean_rows)
    pair_rows = read_csv(PAIR_CSV)
    phase_pairs = parse_phase_pairs()
    phase_rows = phase_summary()
    oracles = build_oracles(pair_rows, medians, phase_pairs)
    make_figures(pair_rows, phase_rows, phase_pairs, oracles)
    summary = {
        "clean_median_seconds": medians,
        "clean_restarts": {d: sum(r["dataset"] == d for r in clean_rows) for d in medians},
        "all_generation_outputs_match_clean": all(r["matches_clean_reference"] for r in quality),
        "pairwise_rows": len(pair_rows),
        "phase_pair_rows": len(phase_pairs),
        "strongest_direct_request_percent": max(float(r["direct_request_e2e_percent"]) for r in oracles),
        "strongest_service_upper_percent": max(float(r["service_throughput_upper_percent"] or 0) for r in oracles),
        "strongest_max_every_wave_service_upper_percent": max(float(r["max_every_wave_service_upper_percent"] or 0) for r in oracles),
        "decision": "NO-OVERLAP-SIGNAL",
    }
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
