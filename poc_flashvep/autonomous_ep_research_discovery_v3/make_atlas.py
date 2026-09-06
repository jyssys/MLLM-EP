#!/usr/bin/env python3
"""Build compact discovery-sprint artifacts from fresh live traces.

This is intentionally descriptive: no latency or utilization values are
invented, and warmup rows are excluded when ``atlas_measured`` is present.
"""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "poc_flashvep/deepep_revalidation/results/autonomous_ep_research_discovery_v3_20260906_172149"
OUT = ROOT / "discovery_atlas"
# Completed fresh runs.  The two explicitly aborted diagnostics are kept on
# disk but intentionally excluded from the primary atlas; they are listed in
# the manifest and experiment log instead of being silently treated as data.
TRACES = [
    "live_mixed_c2", "live_mixed_c4", "live_mixed_c8",
    "live_mixed_c8_mbt4096", "live_high_c16", "live_long_text_c8",
    "live_text_c4", "live_text_c8", "live_text_c8_rep2",
    "live_text_c16", "live_vision_hi_c8",
    # Shape-state controls collected after the initial atlas: deliberately
    # retain them as separate trace labels so the transition effect is
    # inspectable rather than averaged away.
    "live_text_c8_telemetry", "live_text_c8_warmup_text",
    "live_text_to_vision_c8", "live_alternating_text_vision_c8",
    "live_alternating_text_shapes_c8",
    "live_vision_hi_c8_warmup_hi", "live_alternating_vision_shapes_c8",
    "live_text_c8_warmup_text_long", "live_text_to_vision_c8_telemetry",
]

def q(xs, p):
    xs = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs: return None
    i=(len(xs)-1)*p; lo=math.floor(i); hi=math.ceil(i)
    return xs[lo] if lo==hi else xs[lo]+(xs[hi]-xs[lo])*(i-lo)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]; summaries=[]
    for name in TRACES:
        p = ROOT/name/"atlas_measured/logical_invocations.csv"
        if not p.exists(): p = ROOT/name/"atlas/logical_invocations.csv"
        if not p.exists(): continue
        with p.open() as f:
            for r in csv.DictReader(f):
                r["trace"] = name
                for k in ("M","cuda_ms","wall_ms","active_experts","rank_max_mean","rank_cv","expert_cv","fanout_mean","fanout_f4","deepep_layout_ms","deepep_dispatch_ms","expert_ms","deepep_combine_ms","deepep_event_wait_ms"):
                    r[k]=float(r[k] or 0)
                rows.append(r)
        smp=ROOT/name/"atlas_measured/atlas_summary.json"
        if not smp.exists(): smp=ROOT/name/"atlas/atlas_summary.json"
        if smp.exists():
            s=json.loads(smp.read_text()); summaries.append({"trace":name, **s.get("e2e",{}), "logical_rows":s.get("logical_rows"), "stage_summary":s.get("stage_summary",{})})
    with (OUT/"logical_invocations_all.csv").open("w",newline="") as f:
        fields=["trace"]+[k for k in rows[0] if k!="trace"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    # phase totals and regime descriptors
    phase=[]
    for trace in sorted({r["trace"] for r in rows}):
        for ph in sorted({r["phase"] for r in rows if r["trace"]==trace}):
            z=[r for r in rows if r["trace"]==trace and r["phase"]==ph]
            total=sum(r["cuda_ms"] for r in z) or 1.0
            phase.append({"trace":trace,"phase":ph,"n":len(z),"M_p50":q([r["M"] for r in z],.5),"T_MoE_p50_ms":q([r["cuda_ms"] for r in z],.5),"T_MoE_p90_ms":q([r["cuda_ms"] for r in z],.9),"T_MoE_p99_ms":q([r["cuda_ms"] for r in z],.99),"dispatch_share_pct":100*sum(r["deepep_dispatch_ms"] for r in z)/total,"expert_share_pct":100*sum(r["expert_ms"] for r in z)/total,"combine_share_pct":100*sum(r["deepep_combine_ms"] for r in z)/total,"wait_share_pct":100*sum(r["deepep_event_wait_ms"] for r in z)/total,"fanout_p50":q([r["fanout_mean"] for r in z],.5)})
    with (OUT/"phase_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(phase[0])); w.writeheader(); w.writerows(phase)
    (OUT/"e2e_summary.json").write_text(json.dumps(summaries,indent=2))
    # A conservative latency-mass proxy: use measured logical MoE time only,
    # and label it explicitly as stage-mass, not an E2E speedup claim.
    lines=["# E2E latency mass atlas (fresh live traces)","", "All rows below exclude warmup when the measured atlas exists. The stage mass is the sum of rank-collapsed, layer-local CUDA intervals; it is not treated as request-level E2E time because request IDs are only wave labels.","", "| trace | phase | n | M p50 | T_MoE p50/p90/p99 ms | dispatch % | expert % | combine % | wait % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in phase: lines.append(f"| {x['trace']} | {x['phase']} | {x['n']} | {x['M_p50']:.0f} | {x['T_MoE_p50_ms']:.3f}/{x['T_MoE_p90_ms']:.3f}/{x['T_MoE_p99_ms']:.3f} | {x['dispatch_share_pct']:.1f} | {x['expert_share_pct']:.1f} | {x['combine_share_pct']:.1f} | {x['wait_share_pct']:.1f} |")
    lines += ["", "## Direct request observations", ""]
    def fmt(v): return "NA" if v is None else f"{float(v):.2f}"
    for s in summaries: lines.append(f"- {s['trace']}: n={s.get('n')} E2E p50/p90/p99={fmt(s.get('p50_ms'))}/{fmt(s.get('p90_ms'))}/{fmt(s.get('p99_ms'))} ms")
    (OUT/"E2E_LATENCY_ATLAS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"rows":len(rows),"phase_rows":len(phase),"traces":len(summaries),"out":str(OUT)},indent=2))
if __name__=="__main__": main()
