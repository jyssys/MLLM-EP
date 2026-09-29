#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poc_mllm_ep_communication_motivation.contract import read_json


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.3f}%"


def ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f} ms"


def percent_ci(result: dict[str, Any]) -> str:
    lo, hi = result["ci95"]
    return f"{result['mean_percent']:.3f}% (95% CI {lo:.3f}%, {hi:.3f}%)"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", default="CURRENT_BRANCH_HEAD_AT_REPORT_GENERATION")
    args = parser.parse_args()
    q1 = read_json(args.analysis / "q1_stage_breakdown.json")
    q2 = read_json(args.analysis / "q2_modality_routes.json")
    q3 = read_json(args.analysis / "q3_interference.json")
    verdict = read_json(args.analysis / "final_verdict.json")
    audit = read_json(args.audit)
    args.output.mkdir(parents=True, exist_ok=True)
    write(
        args.output / "baseline_readiness.md",
        f"""# Baseline readiness

Status: **{audit["status"]}**

- Model revision: `{audit["model_revision"]}`; BF16; shapes `{json.dumps(audit["shapes"], sort_keys=True)}`.
- Physical GPUs: 2,3,6,7; topology `{audit["topology"]}`; UUIDs are retained in the machine-readable audit.
- Backend: TRUE sparse EP4 with `deepep_high_throughput`, linear expert placement, 32 experts/rank.
- Primary controls: chunked prefill off; `max_num_batched_tokens=20000`; `max_num_seqs=5`; DBO off; EPLB/redundancy off; prefix cache off; eager execution.
- Actual DeepEP prepare/finalize and sparse-expert calls are timed with deferred CUDA events. Logical BF16 payload accounting follows the installed backend's token-destination dedup contract.
- This is a motivation study; no communication method, scheduler policy, or speedup implementation is included.
""",
    )
    pre, dec = q1["prefill"], q1["decode"]
    write(
        args.output / "q1_communication_share.md",
        f"""# Q1 — DeepEP communication share

Measured on TRUE TP2-DP2-EP4. For each layer invocation, the component values are taken from the rank with maximum measured MoE-stage latency, so the reported shares remain a consistent critical-rank decomposition.

| Phase | Dispatch mean / share | Expert mean / share | Combine mean / share | Communication share |
|---|---:|---:|---:|---:|
| Visual prefill | {ms(pre["per_invocation_ms"]["dispatch_ms"]["mean"])} / {pct(pre["shares"]["dispatch"])} | {ms(pre["per_invocation_ms"]["expert_execution_ms"]["mean"])} / {pct(pre["shares"]["expert_execution"])} | {ms(pre["per_invocation_ms"]["combine_ms"]["mean"])} / {pct(pre["shares"]["combine"])} | {pct(pre["shares"]["communication"])} |
| B8 decode | {ms(dec["per_invocation_ms"]["dispatch_ms"]["mean"])} / {pct(dec["shares"]["dispatch"])} | {ms(dec["per_invocation_ms"]["expert_execution_ms"]["mean"])} / {pct(dec["shares"]["expert_execution"])} | {ms(dec["per_invocation_ms"]["combine_ms"]["mean"])} / {pct(dec["shares"]["combine"])} | {pct(dec["shares"]["communication"])} |

Verdict: **{q1["verdict"]}**.

The component trace and clean serving-wall runs are separate. Deferred event synchronization occurs after a sampled 48-layer boundary, not in each layer's measured interval.
""",
    )
    lines = [
        "# Q2 — Visual contribution to prefill EP communication",
        "",
        "The direct byte view uses actual mixed-prefill routes. Physical replays use frozen full-route expert IDs and owner ranks; visual-only plus text-only latency is deliberately not added to reconstruct the nonlinear full collective.",
        "",
        "| Size | Visual tokens (prior processor audit) | Visual token share | Visual dispatch-byte share | Full dispatch | Visual-only dispatch | Text-only dispatch | Full combine | Visual-only combine | Text-only combine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_cases = sorted(
        q2["cases"].values(), key=lambda case: case["request"]["visual_token_count"]
    )
    for case in ordered_cases:
        request = case["request"]
        direct = case["direct_mixed_accounting"]
        conditions = case["conditions"]
        lines.append(
            "| {size} | {vt} | {tf} | {df} | {fd} | {vd} | {td} | {fc} | {vc} | {tc} |".format(
                size=request["size_class"],
                vt=request["visual_token_count"],
                tf=pct(direct["visual_token_fraction"]),
                df=pct(direct["visual_dispatch_byte_fraction"]),
                fd=ms(
                    conditions["FULL_REPLAY"]["per_invocation_ms"]["dispatch_ms"][
                        "mean"
                    ]
                ),
                vd=ms(
                    conditions["VISUAL_ONLY"]["per_invocation_ms"]["dispatch_ms"][
                        "mean"
                    ]
                ),
                td=ms(
                    conditions["TEXT_ONLY"]["per_invocation_ms"]["dispatch_ms"]["mean"]
                ),
                fc=ms(
                    conditions["FULL_REPLAY"]["per_invocation_ms"]["combine_ms"]["mean"]
                ),
                vc=ms(
                    conditions["VISUAL_ONLY"]["per_invocation_ms"]["combine_ms"]["mean"]
                ),
                tc=ms(
                    conditions["TEXT_ONLY"]["per_invocation_ms"]["combine_ms"]["mean"]
                ),
            )
        )
    lines += [
        "",
        f"Verdict: **{q2['verdict']}**.",
        "",
        "Bytes are logical BF16 activation payload after token-destination dedup; metadata/control bytes are not mislabeled as measured transport bytes. Dispatch/combine latencies are physical CUDA-event measurements.",
    ]
    lines += [
        "",
        "## Per-rank mixed-route traffic",
        "",
        "The following values sum 48 layers and all source shards for measured replicate 0.",
        "",
        "| Size | Modality | Rank 0 | Rank 1 | Rank 2 | Rank 3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in ordered_cases:
        direct = case["direct_mixed_accounting"]
        for modality in ("visual", "text", "special_other"):
            ranks = direct["per_rank"][modality]["dispatch_payload_bytes_per_rank"]
            lines.append(
                f"| {case['request']['size_class']} | {modality} | "
                + " | ".join(str(value) for value in ranks)
                + " |"
            )
    write(args.output / "q2_visual_communication_share.md", "\n".join(lines))
    stage = q3["trace_injection_stage_inflation"]
    peaks = q3["paired_injection_window_peak_inflation"]
    overhead = q3.get("instrumentation_overhead_audit")
    overhead_line = (
        "Not run"
        if overhead is None
        else f"{overhead['paired_overhead']['mean_percent']:.3f}% "
        f"({overhead['status']}; same-process paired wrapper bypass)"
    )
    trace_stage = q3["trace_injection_stage"]
    write(
        args.output / "q3_prefill_decode_interference.md",
        f"""# Q3 — Prefill/decode interference

Resident decode is global B8 fixed64. A prefill is injected after 16 generated tokens. Clean TPOT uses homogeneous condition waves and the maximum DP caller wall per engine iteration. Trace runs are separate and describe the shared DeepEP collective experienced by the resident decode; they do not invent a row-wise latency split.

| Condition | Mean ITL | Median ITL | P90 ITL | Maximum stall | Paired mean inflation |
|---|---:|---:|---:|---:|---:|
| Decode only | {ms(q3["itl_ms"]["DECODE_ONLY"]["mean"])} | {ms(q3["itl_ms"]["DECODE_ONLY"]["median"])} | {ms(q3["itl_ms"]["DECODE_ONLY"]["p90"])} | {ms(q3["itl_ms"]["DECODE_ONLY"]["max"])} | — |
| Ordinary text prefill | {ms(q3["itl_ms"]["TEXT_PREFILL"]["mean"])} | {ms(q3["itl_ms"]["TEXT_PREFILL"]["median"])} | {ms(q3["itl_ms"]["TEXT_PREFILL"]["p90"])} | {ms(q3["itl_ms"]["TEXT_PREFILL"]["max"])} | {q3["paired_tpot_inflation"]["TEXT_PREFILL"]["mean_percent"]:.3f}% |
| Large visual prefill | {ms(q3["itl_ms"]["VISUAL_PREFILL"]["mean"])} | {ms(q3["itl_ms"]["VISUAL_PREFILL"]["median"])} | {ms(q3["itl_ms"]["VISUAL_PREFILL"]["p90"])} | {ms(q3["itl_ms"]["VISUAL_PREFILL"]["max"])} | {q3["paired_tpot_inflation"]["VISUAL_PREFILL"]["mean_percent"]:.3f}% |
| Token-matched text prefill | {ms(q3["itl_ms"]["TOKEN_MATCHED_TEXT"]["mean"])} | {ms(q3["itl_ms"]["TOKEN_MATCHED_TEXT"]["median"])} | {ms(q3["itl_ms"]["TOKEN_MATCHED_TEXT"]["p90"])} | {ms(q3["itl_ms"]["TOKEN_MATCHED_TEXT"]["max"])} | {q3["paired_tpot_inflation"]["TOKEN_MATCHED_TEXT"]["mean_percent"]:.3f}% |

## Actual mixed-iteration MoE stage

| Condition | Dispatch mean/layer | Expert mean/layer | Combine mean/layer | MoE mean/layer | Communication inflation | Expert inflation |
|---|---:|---:|---:|---:|---:|---:|
| Decode-only matched reference | {ms(trace_stage["DECODE_ONLY"]["per_invocation_ms"]["dispatch_ms"]["mean"])} | {ms(trace_stage["DECODE_ONLY"]["per_invocation_ms"]["expert_execution_ms"]["mean"])} | {ms(trace_stage["DECODE_ONLY"]["per_invocation_ms"]["combine_ms"]["mean"])} | {ms(trace_stage["DECODE_ONLY"]["per_invocation_ms"]["moe_ms"]["mean"])} | — | — |
| Ordinary text prefill | {ms(trace_stage["TEXT_PREFILL"]["per_invocation_ms"]["dispatch_ms"]["mean"])} | {ms(trace_stage["TEXT_PREFILL"]["per_invocation_ms"]["expert_execution_ms"]["mean"])} | {ms(trace_stage["TEXT_PREFILL"]["per_invocation_ms"]["combine_ms"]["mean"])} | {ms(trace_stage["TEXT_PREFILL"]["per_invocation_ms"]["moe_ms"]["mean"])} | {stage["TEXT_PREFILL"]["communication_inflation_percent"]:.3f}% | {stage["TEXT_PREFILL"]["expert_execution_ms"]:.3f}% |
| Large visual prefill | {ms(trace_stage["VISUAL_PREFILL"]["per_invocation_ms"]["dispatch_ms"]["mean"])} | {ms(trace_stage["VISUAL_PREFILL"]["per_invocation_ms"]["expert_execution_ms"]["mean"])} | {ms(trace_stage["VISUAL_PREFILL"]["per_invocation_ms"]["combine_ms"]["mean"])} | {ms(trace_stage["VISUAL_PREFILL"]["per_invocation_ms"]["moe_ms"]["mean"])} | {stage["VISUAL_PREFILL"]["communication_inflation_percent"]:.3f}% | {stage["VISUAL_PREFILL"]["expert_execution_ms"]:.3f}% |
| Token-matched text prefill | {ms(trace_stage["TOKEN_MATCHED_TEXT"]["per_invocation_ms"]["dispatch_ms"]["mean"])} | {ms(trace_stage["TOKEN_MATCHED_TEXT"]["per_invocation_ms"]["expert_execution_ms"]["mean"])} | {ms(trace_stage["TOKEN_MATCHED_TEXT"]["per_invocation_ms"]["combine_ms"]["mean"])} | {ms(trace_stage["TOKEN_MATCHED_TEXT"]["per_invocation_ms"]["moe_ms"]["mean"])} | {stage["TOKEN_MATCHED_TEXT"]["communication_inflation_percent"]:.3f}% | {stage["TOKEN_MATCHED_TEXT"]["expert_execution_ms"]:.3f}% |

Inflations use the decode-only trace at each condition's physically observed mixed iteration (18 for text controls, 19 for visual), not an assumed caller-clock alignment.

## Injected prefill traffic

| Condition | Prompt tokens/layer | Routed pairs (48 layers) | Logical dispatch bytes | Peak-rank dispatch bytes |
|---|---:|---:|---:|---:|
| Ordinary text | {q3["injected_prefill_route_summary"]["TEXT_PREFILL"]["injected_tokens_per_layer"]} | {q3["injected_prefill_route_summary"]["TEXT_PREFILL"]["token_expert_pairs_across_48_layers"]} | {q3["injected_prefill_route_summary"]["TEXT_PREFILL"]["logical_dispatch_payload_bytes_across_48_layers"]} | {q3["injected_prefill_route_summary"]["TEXT_PREFILL"]["peak_rank_dispatch_payload_bytes_across_48_layers"]} |
| Large visual | {q3["injected_prefill_route_summary"]["VISUAL_PREFILL"]["injected_tokens_per_layer"]} ({q3["injected_prefill_route_summary"]["VISUAL_PREFILL"]["visual_tokens_per_layer"]} visual) | {q3["injected_prefill_route_summary"]["VISUAL_PREFILL"]["token_expert_pairs_across_48_layers"]} | {q3["injected_prefill_route_summary"]["VISUAL_PREFILL"]["logical_dispatch_payload_bytes_across_48_layers"]} | {q3["injected_prefill_route_summary"]["VISUAL_PREFILL"]["peak_rank_dispatch_payload_bytes_across_48_layers"]} |
| Token-matched text | {q3["injected_prefill_route_summary"]["TOKEN_MATCHED_TEXT"]["injected_tokens_per_layer"]} | {q3["injected_prefill_route_summary"]["TOKEN_MATCHED_TEXT"]["token_expert_pairs_across_48_layers"]} | {q3["injected_prefill_route_summary"]["TOKEN_MATCHED_TEXT"]["logical_dispatch_payload_bytes_across_48_layers"]} | {q3["injected_prefill_route_summary"]["TOKEN_MATCHED_TEXT"]["peak_rank_dispatch_payload_bytes_across_48_layers"]} |

Verdict: **{q3["verdict"]}**.

The single injected burst is diluted in the 63-step aggregate TPOT. The primary interruption statistic is therefore the common five-iteration response-window maximum (caller iterations 16–20, covering the measured external-DP async lag): ordinary text {percent_ci(peaks["TEXT_PREFILL"])}, token-matched text {percent_ci(peaks["TOKEN_MATCHED_TEXT"])}, and visual prefill {percent_ci(peaks["VISUAL_PREFILL"])}. Visual prefill also raises measured shared-collective communication by {stage["VISUAL_PREFILL"]["communication_inflation_percent"]:.3f}% at the actual mixed iteration, versus {stage["TOKEN_MATCHED_TEXT"]["communication_inflation_percent"]:.3f}% for token-matched text.

The token-matched control determines whether interference is visual-specific or generic to a same-sized prefill burst. This report does not infer communication causality merely from a slow visual prefill. Instrumentation bookkeeping overhead: {overhead_line}.
""",
    )
    high = q2["cases"][q2["primary_high_case"]]
    hs = high["conditions"]
    final = f"""# Final summary

The bounded study measured standalone communication share, exact visual-route payload, frozen-route isolated communication cost, and concurrent decode interference. It did **not** implement communication-only chunking.

GPU_SET: 2,3,6,7

Q1_PREFILL_DISPATCH_SHARE: {pct(pre["shares"]["dispatch"])}
Q1_PREFILL_COMBINE_SHARE: {pct(pre["shares"]["combine"])}
Q1_PREFILL_COMM_SHARE: {pct(pre["shares"]["communication"])}
Q1_PREFILL_EXPERT_SHARE: {pct(pre["shares"]["expert_execution"])}

Q1_DECODE_DISPATCH_SHARE: {pct(dec["shares"]["dispatch"])}
Q1_DECODE_COMBINE_SHARE: {pct(dec["shares"]["combine"])}
Q1_DECODE_COMM_SHARE: {pct(dec["shares"]["communication"])}
Q1_DECODE_EXPERT_SHARE: {pct(dec["shares"]["expert_execution"])}
Q1_VERDICT: {q1["verdict"]}

Q2_VISUAL_TOKEN_FRACTION: {pct(high["direct_mixed_accounting"]["visual_token_fraction"])}
Q2_VISUAL_DISPATCH_BYTE_FRACTION: {pct(high["direct_mixed_accounting"]["visual_dispatch_byte_fraction"])}
Q2_VISUAL_COMBINE_BYTE_FRACTION: {pct(high["direct_mixed_accounting"]["visual_combine_byte_fraction"])}
Q2_VISUAL_ONLY_DISPATCH_COST: {ms(hs["VISUAL_ONLY"]["per_invocation_ms"]["dispatch_ms"]["mean"])}
Q2_TEXT_ONLY_DISPATCH_COST: {ms(hs["TEXT_ONLY"]["per_invocation_ms"]["dispatch_ms"]["mean"])}
Q2_VISUAL_ONLY_COMBINE_COST: {ms(hs["VISUAL_ONLY"]["per_invocation_ms"]["combine_ms"]["mean"])}
Q2_TEXT_ONLY_COMBINE_COST: {ms(hs["TEXT_ONLY"]["per_invocation_ms"]["combine_ms"]["mean"])}
Q2_VERDICT: {q2["verdict"]}

Q3_DECODE_ONLY_TPOT: {ms(q3["mean_tpot_ms"]["DECODE_ONLY"])}
Q3_TEXT_PREFILL_TPOT: {ms(q3["mean_tpot_ms"]["TEXT_PREFILL"])}
Q3_VISUAL_PREFILL_TPOT: {ms(q3["mean_tpot_ms"]["VISUAL_PREFILL"])}
Q3_TOKEN_MATCHED_TEXT_TPOT: {ms(q3["mean_tpot_ms"]["TOKEN_MATCHED_TEXT"])}
Q3_TEXT_PREFILL_WINDOW_PEAK_INFLATION: {percent_ci(peaks["TEXT_PREFILL"])}
Q3_VISUAL_PREFILL_WINDOW_PEAK_INFLATION: {percent_ci(peaks["VISUAL_PREFILL"])}
Q3_TOKEN_MATCHED_TEXT_WINDOW_PEAK_INFLATION: {percent_ci(peaks["TOKEN_MATCHED_TEXT"])}

Q3_DECODE_DISPATCH_INFLATION_TEXT: {stage["TEXT_PREFILL"]["dispatch_ms"]:.3f}%
Q3_DECODE_DISPATCH_INFLATION_VISUAL: {stage["VISUAL_PREFILL"]["dispatch_ms"]:.3f}%
Q3_DECODE_COMBINE_INFLATION_TEXT: {stage["TEXT_PREFILL"]["combine_ms"]:.3f}%
Q3_DECODE_COMBINE_INFLATION_VISUAL: {stage["VISUAL_PREFILL"]["combine_ms"]:.3f}%
Q3_DECODE_EXPERT_INFLATION_VISUAL: {stage["VISUAL_PREFILL"]["expert_execution_ms"]:.3f}%
Q3_DECODE_MOE_INFLATION_VISUAL: {stage["VISUAL_PREFILL"]["moe_ms"]:.3f}%
INSTRUMENTATION_OVERHEAD: {overhead_line}
Q3_VERDICT: {q3["verdict"]}

OVERALL_VERDICT: {verdict["overall"]}
NEXT_ACTION: {("Proceed to a separate bounded communication-only chunking design PoC." if verdict["overall"] == "COMMUNICATION_ONLY_CHUNKING_MOTIVATED" else "Do not implement communication-only chunking from this evidence; retain the measured characterization.")}
BRANCH: {args.branch}
COMMIT: {args.commit}
"""
    write(args.output / "final_summary.md", final)


if __name__ == "__main__":
    main()
