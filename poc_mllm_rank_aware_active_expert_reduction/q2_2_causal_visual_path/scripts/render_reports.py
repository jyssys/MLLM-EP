from __future__ import annotations

import argparse
from pathlib import Path

from ..contract import read_json


def f(value, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    audit = read_json(args.artifacts / "baseline_audit.json")
    cohort = read_json(args.artifacts / "manifests/frozen_q2_2_cohort.json")
    signals = read_json(args.artifacts / "causal_signal_summary.json")
    gate = read_json(args.artifacts / "causal_gate.json")
    proxy = read_json(args.artifacts / "cheap_proxy_screen.json")
    verdict = read_json(args.artifacts / "final_verdict.json")
    diff = gate["high_low"]["high_minus_low_mean"]
    ci = gate["request_bootstrap"]["ci95"]
    ci_text = "N/A" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"

    write(args.reports / "baseline_readiness.md", f"""
# Q2-2 baseline readiness

Status: **{audit['status']}**

- Frozen model revision: `{audit['model_revision']}`; BF16, d=2048, m=768, E=128, K=8, 48 layers.
- Physical scope: GPUs 2,3,6,7 only; `{audit['topology']}` with `deepep_high_throughput`.
- Prior bounded result is preserved: Q1/Q3 passed while attention-VDS Q2 failed, so Q2-2 tests a new causal target rather than reusing the failed proxy.
- This run is instrumented causal/functional evidence. It is not clean timing and makes no TPOT/E2E claim.
""")
    write(args.reports / "causal_path_evidence.md", f"""
# Q2-2A causal expert-path evidence

The frozen cohort contains {cohort['calibration_requests']} calibration and
{cohort['fresh_heldout_requests']} fresh held-out requests. Every original and
same-domain counterfactual used the identical token-by-token teacher-forced
prefix at the same layer and decode step. The unit is `(request,layer,step,expert)`.

- Held-out paths: {gate['fresh_heldout_paths']}
- Natural counterfactual top-k retention: {signals['counterfactual_topk_retention']:.3%}
- Same-expert logical evaluations: {signals['logical_same_expert_current_weight_evaluations']} BF16 current-weight evaluations; runtime-selected validation max relative-L2 {signals['offline_current_weight_eval_validation']['max_relative_l2']:.6f}.
- PCS vs signed `D_visual` Spearman: **{f(gate['primary_pcs_d_visual_spearman'])}**
- PCS vs generic `D_orig` Spearman: {f(gate['primary_pcs_d_orig_spearman'])}
- High-minus-low mean signed `D_visual`: **{f(diff)}**
- Request-bootstrap 95% CI: **{ci_text}**
- Top-k membership flip rate: {gate['topk_membership_flip_rate']:.3%}
- Exact intervention marker audit: original={gate['marker_audit']['orig']['all_pass']}, counterfactual={gate['marker_audit']['cf']['all_pass']}.
- Free-running counterfactual ablation: N/A; it is not position/token aligned and was not used as causal evidence.

Gate: **{gate['status']}**. The primary gate requires positive Spearman,
positive frozen-bin separation, and a request-bootstrap CI wholly above zero.
Paths absent from counterfactual top-k use the predeclared non-active marginal
convention `g_cf=0, c_cf=0, D_cf=0`; no substitute expert is used.
The frozen original-image continuation can be off-distribution under the
counterfactual image; teacher forcing is required here to isolate visual
content at identical token, position, layer, and expert coordinates.
""")
    if proxy["status"] == "RUN":
        rows = []
        for name, result in proxy["results"].items():
            sep = result["heldout_separation"]["high_minus_low_mean"]
            pci = result["heldout_bootstrap"]["ci95"]
            rows.append(
                f"| {name} | {f(result['heldout_d_visual_spearman'])} | "
                f"{f(result['heldout_d_orig_spearman'])} | {f(sep)} | {pci} |"
            )
        body = "\n".join(rows)
        proxy_text = f"""
# Q2-2B cheap single-pass proxy screen

| proxy | Spearman vs D_visual | Spearman vs D_orig | high-low D_visual | request 95% CI |
|---|---:|---:|---:|---|
{body}

Best modality-aware proxy: `{proxy['best_modality_proxy']}`. Adds held-out
signal beyond router-weight P0: **{proxy['adds_signal_beyond_router']}**.
All lookup tables, fallbacks, and thresholds were fit on calibration only.
"""
    else:
        proxy_text = """
# Q2-2B cheap single-pass proxy screen

Status: **NOT_RUN_Q2_2A_GATE_FAILED**. Per the source-of-truth stop rule, no
proxy formula was promoted after the primary causal premise failed.
"""
    write(args.reports / "cheap_proxy_screen.md", proxy_text)
    router = "N/A"
    best = "NOT_RUN_Q2_2A_GATE_FAILED"
    if proxy["status"] == "RUN":
        p0 = proxy["results"]["proxy_p0_router"]
        router = f"rho={f(p0['heldout_d_visual_spearman'])}, high-low={f(p0['heldout_separation']['high_minus_low_mean'])}"
        key = proxy["best_modality_proxy"]
        row = proxy["results"][key]
        best = f"{key}: rho={f(row['heldout_d_visual_spearman'])}, high-low={f(row['heldout_separation']['high_minus_low_mean'])}"
    write(args.reports / "final_summary.md", f"""
# Q2-2 final summary

## Answers to the five required questions

1. Changing only visual content causally changed same-path routing/contribution measurements (held-out membership flip {gate['topk_membership_flip_rate']:.3%}); this alone is not the functional gate.
2. Greater PCS predicted greater signed visual-specific drop damage: **{gate['status']}** under the joint held-out rule.
3. Generic `D_orig` and signed `D_visual` are reported separately; the verdict is based only on signed `D_visual`.
4. Cheap modality-aware proxy screen: **{proxy['status']}**.
5. Proceed to a vision-protected rank-aware candidate: **{'YES, separately gated' if verdict['verdict'] == 'MLLM_CAUSAL_LINK_AND_PROXY_SUPPORTED' else 'NO'}**.

No TPOT, E2E, throughput, benchmark-quality, production, or EP8-speedup claim is made.

```text
Q2_2_PRIMARY_PCS_DVISUAL_SPEARMAN: {f(gate['primary_pcs_d_visual_spearman'])}
Q2_2_HIGH_MINUS_LOW_DVISUAL: {f(diff)}
Q2_2_HIGH_MINUS_LOW_95CI: {ci_text}
Q2_2_ROUTER_BASELINE: {router}
Q2_2_BEST_MODALITY_PROXY: {best}
Q2_2_FRESH_HELDOUT_REQUESTS: {gate['fresh_heldout_requests']}
Q2_2_VERDICT: {verdict['verdict']}
NEXT_ACTION: {verdict['next_action']}
```
""")

if __name__ == "__main__":
    main()
