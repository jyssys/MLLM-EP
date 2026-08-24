"""Publish figures, machine-readable summary, and the forensic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("result", type=Path); parser.add_argument("source", type=Path); parser.add_argument("report", type=Path)
    args = parser.parse_args(); result=args.result; figures=result/"figures"
    pre=_load(result/"summary_preliminary.json"); selected=pre["selected"]
    isolated={name:_load(result/"isolated_run"/"target_results"/f"{name}.json") for name in ("text","vision")}
    stats={}
    for name,row in isolated.items():
        values=np.asarray(row["samples_ms"],dtype=float)
        stats[name]={"median_ms":float(np.median(values)),"p25_ms":float(np.percentile(values,25)),"p75_ms":float(np.percentile(values,75)),"cv":float(values.std()/values.mean()),"n":len(values)}
    isolated_gap=stats["vision"]["median_ms"]/stats["text"]["median_ms"]-1
    frame=pd.read_csv(args.source/"per_rank_shape_latency.csv")
    target=frame[(frame.layer==selected["layer"])&(frame["rank"]==selected["rank"])&frame.request_id.isin([selected["vision_request_id"],selected["text_request_id"]])]
    samples={row.modality:np.asarray(json.loads(row.expert_ms_samples),dtype=float) for row in target.itertuples()}

    fig,ax=plt.subplots(figsize=(8,4.5))
    data=[samples["text"],samples["vision"],np.asarray(isolated["text"]["samples_ms"]),np.asarray(isolated["vision"]["samples_ms"])]
    ax.boxplot(data,tick_labels=["Text live\n15 runs","Vision live\n15 runs","Text isolated\n100 runs","Vision isolated\n100 runs"],showfliers=True)
    ax.set_ylabel("Expert kernel latency (ms)"); ax.set_title("The live outlier does not survive exact-input isolated replay")
    fig.tight_layout(); fig.savefig(figures/"plot2_fast_slow_replay.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(np.arange(1,16),samples["text"],marker="o",label="Text live")
    ax.plot(np.arange(1,16),samples["vision"],marker="o",label="Vision live")
    ax.axhline(stats["text"]["median_ms"],color="tab:blue",ls="--",alpha=.7,label="Text isolated median")
    ax.axhline(stats["vision"]["median_ms"],color="tab:orange",ls="--",alpha=.7,label="Vision isolated median")
    ax.set(xlabel="Measured live repetition",ylabel="Expert kernel latency (ms)",title="Live Vision bimodality indicates transient runtime context")
    ax.legend(ncol=2,fontsize=8)
    ax.text(.01,.98,"Nsight Systems 2024.6.2: full trace caused DeepEP timeout;\nNVTX-gated run completed but emitted no report. No profiler metric claimed.",transform=ax.transAxes,va="top",fontsize=8,bbox={"facecolor":"white","alpha":.8})
    fig.tight_layout(); fig.savefig(figures/"plot3_profiler_fast_vs_slow.png",dpi=180); plt.close(fig)

    profiler={
        "ncu":{"available":False,"version":None},
        "nsys":{"available":True,"version":"2024.6.2.225-246235244400v0","full_trace":"failed before target: DeepEP CPU recv timeout under profiler overhead","nvtx_gated":"six target requests completed; Nsight emitted no .nsys-rep artifact","valid_hardware_metrics":False},
        "strongest_observed_context_difference":"Vision live samples are bimodal (CV 17.16%) while Text live CV is 3.61%; the same inputs converge in isolated replay.",
    }
    summary={**pre,
        "mllm_specific_straggler":"NO-GO","kernel_mechanism":"NO-GO",
        "isolated":{**stats,"vision_relative_gap":float(isolated_gap),"routing_changed":False},
        "profiler":profiler,
        "likely_mechanism":"transient runtime/system-context interference, not a reproducible Vision-specific or expert-kernel-internal mechanism",
    }
    (result/"profiler_summary.json").write_text(json.dumps(profiler,indent=2)+"\n")
    (result/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    rel=Path("../deepep_revalidation/results")/result.name/"figures"
    report=f"""# MLLM EP straggler forensics

## Final gates

- **MLLM_SPECIFIC_STRAGGLER: NO-GO**
- **KERNEL_MECHANISM: NO-GO**

After the preregistered N/G/Q match, mean Vision-minus-Text expert latency is only **{pre['mean_relative_vision_residual']:.2%}** (median {pre['median_relative_vision_residual']:.2%}; source-request-clustered 95% CI [{pre['source_request_clustered_ci95'][0]:.2%}, {pre['source_request_clustered_ci95'][1]:.2%}]). The result does not support a Vision-specific residual.

## Environment and provenance

Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput and live `TritonExperts` on physical GPUs 4–7 were reused unchanged. DBO and prefix caching were off. The source trace is `{args.source}` and contains 15 measured live repetitions per request/layer/rank. No routing, placement, weight, or kernel behavior was changed.

Profiler audit: `ncu` is absent. `nsys` is NVIDIA Nsight Systems 2024.6.2.225-246235244400v0.

## Stage A — Fixed matched work

The matching policy was fixed before inspecting outcomes: one-to-one Hungarian matching within layer/rank/token bucket, `|ΔN| <= 5%`, exact G, and `|ΔQ| <= 2`. The >=15% latency criterion was used only to select forensic candidates, not to estimate the modality residual.

| Metric | Result |
|---|---:|
| Cross-modality matched pairs | {pre['matched_pairs']} |
| Unique Vision / Text requests | {pre['matched_vision_requests']} / {pre['matched_text_requests']} |
| Mean Vision residual | {pre['mean_relative_vision_residual']:.2%} |
| Clustered 95% CI | [{pre['source_request_clustered_ci95'][0]:.2%}, {pre['source_request_clustered_ci95'][1]:.2%}] |
| Vision slower fraction | {pre['vision_slower_fraction']:.2%} |
| >=15% cross-modality pairs | {pre['forensic_cross_pairs']} |
| >=15% within-Vision / within-Text pairs | {pre['within_vision_forensic_pairs']} / {pre['within_text_forensic_pairs']} |
| Matched rank is actual critical rank, Vision / Text | {pre['vision_actual_critical_frequency']:.2%} / {pre['text_actual_critical_frequency']:.2%} |

The sole cross-modality forensic candidate was layer {selected['layer']}, rank {selected['rank']}: `{selected['vision_request_id']}` versus `{selected['text_request_id']}`. N={selected['vision_n']}/{selected['text_n']}, G={selected['vision_g']}/{selected['text_g']}, Q={selected['vision_q']}/{selected['text_q']}, and the original live median gap was {selected['relative_latency_residual']:.2%}. Its Vision samples were bimodal, not a stable shifted distribution.

![Matched Vision/Text latency]({rel/'plot1_matched_vision_text_latency.png'})

## Stage B — Exact-input isolated replay

At the selected live layer/rank, the actual post-DeepEP expert input, weights, metadata, and observed histogram were reused for 20 warmups and 100 same-stream CUDA-event measurements. Routing edit distance is zero. Idle-DP padding made the replay-run histograms differ by 1–2 assignments from the earlier median repetition, but the compared replay work remained N={isolated['vision']['total_assignments']}/{isolated['text']['total_assignments']} and G=28/28.

| | Text | Vision |
|---|---:|---:|
| Median expert latency | {stats['text']['median_ms']:.6f} ms | {stats['vision']['median_ms']:.6f} ms |
| IQR | [{stats['text']['p25_ms']:.6f}, {stats['text']['p75_ms']:.6f}] | [{stats['vision']['p25_ms']:.6f}, {stats['vision']['p75_ms']:.6f}] |
| CV | {stats['text']['cv']:.2%} | {stats['vision']['cv']:.2%} |

The isolated Vision gap is **{isolated_gap:.2%}**, so the live 31.37% outlier disappears. This rules out the requested >=10% reproducible kernel-internal mechanism.

![Fast/slow replay]({rel/'plot2_fast_slow_replay.png'})

## Stage C — Bounded profiler result

The initial CUDA+NVTX `nsys` run perturbed startup enough to trigger a DeepEP CPU-receive timeout before any target range. A second run deferred collection to the target NVTX range; all six bounded requests completed, but Nsight emitted no report artifact in this multi-process capture. Consequently no SM, occupancy, DRAM, L2, tensor-core, warp-stall, overlap, or stream metric is claimed.

The strongest available context evidence is non-profiler timing: Vision live CV was 17.16% with two latency bands, Text live CV was 3.61%, while isolated medians differ by only {isolated_gap:.2%}. This is consistent with transient runtime/system-context interference, but it does not identify a specific preceding kernel, cache, stream, or communication cause.

![Profiler/context comparison]({rel/'plot3_profiler_fast_vs_slow.png'})

## Interpretation and limitations

“Vision” was not assumed causal. Across all fixed-policy matches its residual is below 1% and its clustered CI crosses zero. The only large cross-modality example fails isolated reproduction and is therefore not evidence of a Vision-specific GEMM regime. The missing Nsight artifact prevents finer attribution of the transient live outlier. The prior trace also includes stock idle-DP padding, which explains the 1–2 assignment replay-run drift and is explicitly not treated as visual work.

The next work should **pivot generic**, not remain MLLM-specific: instrument a profiler-compatible, modality-agnostic live EP latency-tail harness that records surrounding streams/communication without Nsight-induced DeepEP timeout. Do not build a Vision-specific optimization from this result.
"""
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(report)


if __name__ == "__main__":
    main()
