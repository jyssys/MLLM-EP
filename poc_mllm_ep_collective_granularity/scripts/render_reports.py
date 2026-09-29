#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from poc_mllm_ep_collective_granularity.contract import atomic_json, read_json


def number(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def ci(metric: dict[str, Any], suffix: str = "") -> str:
    return (
        f"{number(metric['mean'])}{suffix} "
        f"[{number(metric['ci95'][0])}, {number(metric['ci95'][1])}]"
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def figures(chunks: dict[str, Any], layered: dict[str, Any], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    full = chunks["FULL"]
    curves = {
        "visual": full["clean"]["event_aligned_vsel_curve_ms"],
        "matched text": full["matched_text_clean"]["event_aligned_vsel_curve_ms"],
    }
    for label, curve in curves.items():
        x = sorted(map(int, curve))
        plt.plot(x, [curve[str(v)]["mean"] for v in x], marker="o", label=label)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("iteration relative to first physical mixed iteration")
    plt.ylabel("paired VSEL (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "event_aligned_vsel.png", dpi=180)
    plt.close()

    labels = ["FULL", "CP2048", "CP1024", "CP512", "CP256"]
    mncd = [chunks[x]["trace"]["mncd_dispatch_ms"]["median"] for x in labels]
    cidd = [chunks[x]["trace"]["cidd_ms_per_layer"]["mean"] for x in labels]
    positions = list(range(len(labels)))
    width = 0.38
    plt.bar([x - width / 2 for x in positions], mncd, width, label="MNCD")
    plt.bar([x + width / 2 for x in positions], cidd, width, label="CIDD/layer")
    plt.xticks(positions, labels)
    plt.ylabel("milliseconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "chunk_comm_tradeoff.png", dpi=180)
    plt.close()

    reload_amp = [chunks[x]["trace"]["expert_reload_amp"]["median"] for x in labels]
    expert = [
        chunks[x]["trace"]["mixed_expert_execution_ms_per_request"]["mean"]
        for x in labels
    ]
    fig, left = plt.subplots()
    left.plot(positions, reload_amp, marker="o", label="ExpertReloadAmp")
    left.set_ylabel("ExpertReloadAmp")
    left.set_xticks(positions, labels)
    right = left.twinx()
    right.plot(positions, expert, marker="s", color="tab:red", label="expert total")
    right.set_ylabel("mixed-iteration expert execution total (ms)")
    fig.tight_layout()
    fig.savefig(root / "chunk_expert_tradeoff.png", dpi=180)
    plt.close(fig)

    plt.bar(
        ["CP512 victim-iteration CIDD", "Layered stage CIDD"],
        [
            chunks["CP512"]["trace"]["cidd_ms_per_victim_iteration"]["mean"],
            layered["cidd_ms_primary"],
        ],
    )
    plt.ylabel("milliseconds per victim scheduling quantum")
    plt.tight_layout()
    plt.savefig(root / "layered_gap.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    chunks = read_json(args.results / "chunk_results.json")
    layered = read_json(args.results / "layered_physical_replay.json")
    moe_replay = read_json(args.results / "moe_only_physical_replay.json")
    verdict = read_json(args.results / "final_verdict.json")
    figures(chunks, layered, args.figures)
    full = chunks["FULL"]
    text = full["matched_text_trace"]
    gates = verdict["gates"]

    write(
        args.reports / "rq1_direct_comm_stall.md",
        f"""# RQ1 — Direct communication-induced decode stall

Evidence is physical TRUE TP2-DP2-EP4 DeepEP on GPUs 2,3,6,7. Clean caller-wall timing and CUDA-event trace are separate. Iteration zero is the first trace-confirmed mixed prefill/decode iteration.

| Metric | Result |
|---|---:|
| Isolated decode dispatch / MoE layer | {number(full["trace"]["decode_isolated_dispatch_ms_per_layer"]["median"])} ms |
| Visual mixed dispatch / MoE layer | {ci(full["trace"]["mixed_dispatch_ms_per_layer"], " ms")} |
| Visual mixed MNCD dispatch | {number(full["trace"]["mncd_dispatch_ms"]["max"])} ms |
| Visual CIDD / MoE layer | {ci(full["trace"]["cidd_ms_per_layer"], " ms")} |
| Visual CIDD / affected victim iteration | {ci(full["trace"]["cidd_ms_per_victim_iteration"], " ms")} |
| Peak VSEL | {ci(full["clean"]["peak_vsel_ms"], " ms")} |
| Peak VSEL percent | {ci(full["clean"]["peak_vsel_percent"], "%")} |
| Decode-only response peak | {number(full["clean"]["decode_only_response_peak_ms"]["mean"])} ms |
| Visual response peak | {number(full["clean"]["condition_response_peak_ms"]["mean"])} ms |
| Response-window peak inflation | {ci(full["clean"]["response_peak_inflation_percent"], "%")} |

`CIDD` is the measured critical-rank completion excess of the resident decode's physical shared DeepEP dispatch relative to its aligned isolated dispatch. No host timestamp is relabeled as collective wait time; CWT is `UNAVAILABLE_NO_RELIABLE_GPU_READY_TIMESTAMP`. MNCD is a non-preemptible communication quantum, not an assertion that a victim waited exactly that duration.

RQ1 verdict: **{"DIRECT_COMM_STALL_CONFIRMED" if gates["G1"] else "NO_REPEATABLE_DIRECT_STALL"}**.
""",
    )

    table = []
    for label in ("FULL", "CP2048", "CP1024", "CP512", "CP256"):
        trace = chunks[label]["trace"]
        clean = chunks[label]["clean"]
        clean_peak = (
            f"{number(clean['response_peak_inflation_percent']['mean'])}%"
            if clean["alignment_status"] == "PASS"
            else clean["alignment_status"]
        )
        table.append(
            f"| {label} | {number(trace['chunk_count']['median'], 1)} | "
            f"{number(trace['mncd_dispatch_ms']['median'])} | "
            f"{number(trace['cidd_ms_per_layer']['mean'])} | "
            f"{clean_peak} | "
            f"{number(trace['logical_dispatch_bytes_mean_replicate'] / 1e6)} | "
            f"{number(trace['expert_reload_amp']['median'])} | "
            f"{number(trace['estimated_expert_weight_bytes_proxy_per_layer']['median'] / 1e9)} | "
            f"{number(trace['mixed_expert_execution_ms_per_request']['mean'])} | "
            f"{number(trace['incremental_expert_execution_ms_per_request']['mean'])} | "
            f"{number(trace['mixed_moe_ms_per_request']['mean'])} | "
            f"{number(trace['injected_prefill_wall_ms']['mean'])} | "
            f"{number(trace['expert_paths_per_active_expert'])} |"
        )
    write(
        args.reports / "rq2_chunked_tradeoff.md",
        """# RQ2 — Chunked-Prefill communication vs MoE efficiency

Each arm uses the same 2,029-token high-resolution visual request and resident fixed64 B8 victim. `ExpertReloadAmp` is structural. Mixed totals sum critical-rank CUDA-event time over every affected chunk/layer; expert excess subtracts aligned resident-decode execution. Estimated weight bytes use the exact loaded BF16 tensor shapes (9,437,184 bytes/expert) and are explicitly not measured HBM traffic.

| Baseline | physical injection chunks | MNCD dispatch ms | CIDD ms/layer | response peak | logical dispatch MB | ExpertReloadAmp | estimated expert-weight GB/layer | mixed expert total ms | expert excess ms | mixed MoE total ms | prefill wall ms | paths/active expert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(table)
        + "\n\nA clean caller-wall response peak is marked `UNRESOLVED_ASYNC_SCHEDULER_DRIFT` "
        "when the first physical mixed iteration moves by more than four caller iterations "
        "across repeats. CUDA-event MNCD/CIDD and internal physical chunk counts remain "
        "available and are not replaced by the unstable caller-wall value."
        + f"\n\nRQ2 verdict: **{'CHUNKED_COMM_VS_MOE_TRADEOFF_CONFIRMED' if gates['G2'] else 'NO_RUNTIME_SUPPORTED_TRADEOFF'}**.\n",
    )

    size_rows = [
        f"| {size} | {value['prompt_tokens']} | "
        f"{number(value['full_prompt_dispatch_ms_mean'])} ms | {value['evidence']} |"
        for size, value in sorted(
            layered.get("visual_size_sweep", {}).items(),
            key=lambda item: item[1]["prompt_tokens"],
        )
    ]
    write(
        args.reports / "rq3_layered_ep_gap.md",
        f"""# RQ3 — Layered Prefill on EP

Baseline type: **{layered["baseline_type"]}**. This is not native end-to-end Layered serving. Every active-layer datum is a physical full-prompt TRUE-EP4 DeepEP call; the contiguous groups are composed according to the pinned upstream scheduler.

The inspected upstream rule selects **{layered["stage_count"]} stages × {layered["layers_per_stage"]} layers** for 2,029 tokens. This contradicts the spec's stated eight-stage expectation; the code's `<=2048` comparison unambiguously selects four.

| Metric | Layered replay | CP512 physical |
|---|---:|---:|
| ExpertReloadAmp | {number(layered["expert_reload_amp"])} | {number(chunks["CP512"]["trace"]["expert_reload_amp"]["median"])} |
| injected expert-execution excess | {number(full["trace"]["incremental_expert_execution_ms_per_request"]["mean"])} ms | {number(chunks["CP512"]["trace"]["incremental_expert_execution_ms_per_request"]["mean"])} ms |
| mixed expert-execution total | {number(full["trace"]["mixed_expert_execution_ms_per_request"]["mean"])} ms | {number(chunks["CP512"]["trace"]["mixed_expert_execution_ms_per_request"]["mean"])} ms |
| injected MoE excess | {number(full["trace"]["incremental_moe_ms_per_request"]["mean"])} ms | {number(chunks["CP512"]["trace"]["incremental_moe_ms_per_request"]["mean"])} ms |
| MNCD dispatch | {number(full["trace"]["mncd_dispatch_ms"]["median"])} ms | {number(chunks["CP512"]["trace"]["mncd_dispatch_ms"]["median"])} ms |
| CIDD | {number(layered["cidd_ms_primary"])} ms/stage | {number(chunks["CP512"]["trace"]["cidd_ms_per_victim_iteration"]["mean"])} ms/victim iteration |
| replay response peak inflation | {number(layered["response_peak_inflation_percent_replay"])}% | measured {number(chunks["CP512"]["clean"]["response_peak_inflation_percent"]["mean"])}% |

Reused physical size sweep:

| visual size | prompt tokens | full-prompt dispatch/layer | evidence |
|---|---:|---:|---|
{chr(10).join(size_rows)}

The comparison uses matched scheduling quanta: a Layered stage contains {layered["layers_per_stage"]} measured full-prompt layer collectives, while CP512 reports the sum across all affected layers in its worst victim iteration. Per-layer CIDD remains available in the machine-readable artifact. RQ3 verdict: **{"LAYERED_LEAVES_EP_COLLECTIVE_GAP" if gates["G3"] else "LAYERED_GAP_NOT_CONFIRMED"}**.
""",
    )

    root = verdict["rq4_root_cause"]
    replay_visual = moe_replay.get("VISUAL_PREFILL_MOE_REPLAY", {})
    replay_text = moe_replay.get("TOKEN_MATCHED_TEXT_MOE_REPLAY", {})
    replay_delta = moe_replay.get("visual_minus_text_paired_ms_per_layer", {}).get(
        "dispatch_ms", {"mean": None, "ci95": [None, None]}
    )
    write(
        args.reports / "rq4_visual_text_rootcause.md",
        f"""# RQ4 — Visual versus token-matched text root cause

The comparison uses 10 paired clean repeats aligned to the trace-confirmed caller-visible mixed iteration, plus separate physical DeepEP traces.

| Metric | Visual 2,029 tokens (1,980 visual) | token-matched text 2,033 tokens |
|---|---:|---:|
| logical dispatch bytes | {number(full["trace"]["logical_dispatch_bytes_mean_replicate"], 0)} | {number(text["logical_dispatch_bytes_mean_replicate"], 0)} |
| routing HHI | {number(full["trace"]["routing_hhi"], 6)} | {number(text["routing_hhi"], 6)} |
| MNCD dispatch | {number(full["trace"]["mncd_dispatch_ms"]["median"])} ms | {number(text["mncd_dispatch_ms"]["median"])} ms |
| CIDD | {ci(full["trace"]["cidd_ms_per_layer"], " ms/layer")} | {ci(text["cidd_ms_per_layer"], " ms/layer")} |
| response peak inflation | {ci(full["clean"]["response_peak_inflation_percent"], "%")} | {ci(full["matched_text_clean"]["response_peak_inflation_percent"], "%")} |
| exact-hidden MoE-only replay dispatch/layer | {ci(replay_visual.get("dispatch_ms_per_layer", {"mean": None, "ci95": [None, None]}), " ms")} | {ci(replay_text.get("dispatch_ms_per_layer", {"mean": None, "ci95": [None, None]}), " ms")} |
| exact-hidden MoE-only replay MoE/layer | {ci(replay_visual.get("moe_ms_per_layer", {"mean": None, "ci95": [None, None]}), " ms")} | {ci(replay_text.get("moe_ms_per_layer", {"mean": None, "ci95": [None, None]}), " ms")} |

Exact-hidden replay status: `{moe_replay["status"]}`. The paired visual-minus-text replay dispatch difference is {ci(replay_delta, " ms/layer")}. Each replay clones the actual current layer hidden tensor, top-k IDs, and weights before the native call, executes one additional physical TRUE-EP4 DeepEP MoE call, discards its output, and leaves model semantics unchanged.

Root-cause classification: **{root}**. A V-CACHED end-to-end arm was not fabricated because the installed runtime exposes no validated external multimodal-embedding cache contract. Request preprocessing is timed before admission and is outside VSEL; vision-encoder work after admission remains part of the clean mixed iteration.

G4: **{verdict["g4_label"]}**.
""",
    )

    overall = verdict["overall_verdict"]
    final = f"""# Final summary — MLLM MoE EP collective-granularity gap

All primary timings are physical TRUE EP4 measurements. The final communication-only slicing method was not implemented.

```text
GPU_SET: 2,3,6,7

UPSTREAM_LAYERED_REPO: https://github.com/scale-snu/layered-prefill
UPSTREAM_LAYERED_COMMIT: 053f80e5201a7c0ab56e468e3d578907a2ca9cc3
LAYERED_BASELINE_TYPE: LAYERED_EP_PHYSICAL_REPLAY

RQ1_DECODE_ISOLATED_DISPATCH_MS: {number(full["trace"]["decode_isolated_dispatch_ms_per_layer"]["median"])}
RQ1_VISUAL_MIXED_DISPATCH_MS: {number(full["trace"]["mixed_dispatch_ms_per_layer"]["mean"])} per layer
RQ1_VISUAL_RW_PEAK_ITL_MS: {number(full["clean"]["condition_response_peak_ms"]["mean"])}
RQ1_DECODE_ONLY_RW_PEAK_ITL_MS: {number(full["clean"]["decode_only_response_peak_ms"]["mean"])}
RQ1_VISUAL_RW_PEAK_INFLATION: {number(full["clean"]["response_peak_inflation_percent"]["mean"])}%
RQ1_VISUAL_CIDD_MS: {number(full["trace"]["cidd_ms_per_layer"]["mean"])} per layer
RQ1_VISUAL_CIDD_VICTIM_ITERATION_MS: {number(full["trace"]["cidd_ms_per_victim_iteration"]["mean"])}
RQ1_VISUAL_MNCD_DISPATCH_MS: {number(full["trace"]["mncd_dispatch_ms"]["max"])}
RQ1_VERDICT: {"DIRECT_COMM_STALL_CONFIRMED" if gates["G1"] else "NO_REPEATABLE_DIRECT_STALL"}

RQ2_CP_FULL_MNCD_MS: {number(full["trace"]["mncd_dispatch_ms"]["median"])}
RQ2_CP_1024_MNCD_MS: {number(chunks["CP1024"]["trace"]["mncd_dispatch_ms"]["median"])}
RQ2_CP_512_MNCD_MS: {number(chunks["CP512"]["trace"]["mncd_dispatch_ms"]["median"])}
RQ2_CP_256_MNCD_MS: {number(chunks["CP256"]["trace"]["mncd_dispatch_ms"]["median"])}
RQ2_CP_FULL_EXPERT_RELOAD_AMP: {number(full["trace"]["expert_reload_amp"]["median"])}
RQ2_CP_512_EXPERT_RELOAD_AMP: {number(chunks["CP512"]["trace"]["expert_reload_amp"]["median"])}
RQ2_CP_256_EXPERT_RELOAD_AMP: {number(chunks["CP256"]["trace"]["expert_reload_amp"]["median"])}
RQ2_CP_FULL_MIXED_EXPERT_TOTAL_MS: {number(full["trace"]["mixed_expert_execution_ms_per_request"]["mean"])}
RQ2_CP_512_MIXED_EXPERT_TOTAL_MS: {number(chunks["CP512"]["trace"]["mixed_expert_execution_ms_per_request"]["mean"])}
RQ2_VERDICT: {"CHUNKED_COMM_VS_MOE_TRADEOFF_CONFIRMED" if gates["G2"] else "NO_RUNTIME_SUPPORTED_TRADEOFF"}

RQ3_LAYERED_STAGE_COUNT: {layered["stage_count"]}
RQ3_LAYERED_EXPERT_RELOAD_AMP: {number(layered["expert_reload_amp"])}
RQ3_LAYERED_INCREMENTAL_EXPERT_MS: {number(full["trace"]["incremental_expert_execution_ms_per_request"]["mean"])}
RQ3_LAYERED_MNCD_DISPATCH_MS: {number(full["trace"]["mncd_dispatch_ms"]["median"])}
RQ3_LAYERED_CIDD_MS: {number(layered["cidd_ms_primary"])} per stage
RQ3_LAYERED_RW_PEAK_INFLATION: {number(layered["response_peak_inflation_percent_replay"])}% replay
RQ3_VERDICT: {"LAYERED_LEAVES_EP_COLLECTIVE_GAP" if gates["G3"] else "LAYERED_GAP_NOT_CONFIRMED"}

RQ4_VISUAL_TOKENS: 1980 / 2029 total
RQ4_MATCHED_TEXT_TOKENS: 2033
RQ4_VISUAL_LOGICAL_BYTES: {number(full["trace"]["logical_dispatch_bytes_mean_replicate"], 0)}
RQ4_MATCHED_TEXT_LOGICAL_BYTES: {number(text["logical_dispatch_bytes_mean_replicate"], 0)}
RQ4_VISUAL_RW_PEAK_INFLATION: {number(full["clean"]["response_peak_inflation_percent"]["mean"])}%
RQ4_MATCHED_TEXT_RW_PEAK_INFLATION: {number(full["matched_text_clean"]["response_peak_inflation_percent"]["mean"])}%
RQ4_VISUAL_CIDD_MS: {number(full["trace"]["cidd_ms_per_layer"]["mean"])} per layer
RQ4_MATCHED_TEXT_CIDD_MS: {number(text["cidd_ms_per_layer"]["mean"])} per layer
RQ4_MOE_ONLY_REPLAY_STATUS: {moe_replay["status"]}
RQ4_MOE_ONLY_VISUAL_MINUS_TEXT_DISPATCH_MS: {number(replay_delta["mean"])}; CI95=[{number(replay_delta["ci95"][0])},{number(replay_delta["ci95"][1])}]
RQ4_ROOT_CAUSE: {root}
RQ4_VERDICT: {verdict["g4_label"]}

G1_DIRECT_STALL: {"PASS" if gates["G1"] else "FAIL"}
G2_CHUNKED_TRADEOFF: {"PASS" if gates["G2"] else "FAIL"}
G3_LAYERED_EP_GAP: {"PASS" if gates["G3"] else "FAIL"}
G4_MLLM_SPECIFIC: {"PASS" if gates["G4"] else "FAIL"}

OVERALL_VERDICT: {overall}
MLLM_FRAMING: {verdict["g4_label"]}
NEXT_ACTION: {"Proceed to a bounded communication-only chunking implementation PoC." if overall == "PROCEED_COMMUNICATION_ONLY_CHUNKING" else "Do not implement communication-only chunking from this evidence."}
BRANCH: flashvep/mllm-ep-collective-granularity-gap-poc
COMMIT: RECORDED_AFTER_REPORT_GENERATION
```
"""
    write(args.reports / "final_summary.md", final)
    atomic_json(
        args.results / "report_manifest.json",
        {
            "reports": sorted(path.name for path in args.reports.glob("*.md")),
            "figures": sorted(path.name for path in args.figures.glob("*.png")),
            "status": "RENDERED",
        },
    )


if __name__ == "__main__":
    main()
