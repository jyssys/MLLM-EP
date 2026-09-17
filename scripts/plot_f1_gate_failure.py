#!/usr/bin/env python3
"""Create evidence-only figures for the stopped F1 n=32 gate."""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("reports/figures/offline_main_table")
OUT.mkdir(parents=True, exist_ok=True)
BASE = {r["sample_id"]: r for r in map(json.loads, open("artifacts/reproduction/gsm8k_samples.jsonl"))
        if r["sample_id"] < 32}


def rows(directory):
    result = {}
    for path in glob.glob(str(Path(directory) / "generations_*.jsonl")):
        for line in open(path):
            row = json.loads(line); result[row["sample_id"]] = row
    return {key: value for key, value in result.items() if key < 32}


f0 = rows("artifacts/virtual_ep/20260917_selective_refinement/rollouts/p2_confhigh_f0_r75_age1_sanity8")
f1 = rows("artifacts/main_table/f1_gsm8k32")
methods = ["Vanilla", "F0", "F1"]
sets = [BASE, f0, f1]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].bar(methods, [100*np.mean([r["correct"] for r in x.values()]) for x in sets]); axes[0].set_ylabel("GSM8K-32 accuracy (%)")
axes[1].bar(methods, [np.mean([r["nfe"] for r in x.values()]) for x in sets]); axes[1].set_ylabel("mean NFE")
axes[2].bar(methods, [49906, sum(r["active_updates"] for r in f0.values()), sum(r["active_updates"] for r in f1.values())]); axes[2].set_ylabel("W_active")
fig.suptitle("F1 generalization gate (n=32; 128 not run after failure)"); fig.tight_layout(); fig.savefig(OUT/"01_f0_f1_gate32_quality_nfe_work.png",dpi=180); plt.close(fig)

by_iteration = {}
for path in glob.glob("artifacts/main_table/f1_gsm8k32/traces/*.npz"):
    with np.load(path) as z:
        for iteration, masked, active in zip(z["iteration_id"],z["masked"],z["active"]):
            fresh=int((~(masked & ~active)).sum()); reused=int((masked & ~active).sum())
            slot=by_iteration.setdefault(int(iteration),[0,0]);slot[0]+=fresh;slot[1]+=reused
x=sorted(by_iteration);plt.figure(figsize=(7,4));plt.plot(x,[by_iteration[i][0] for i in x],label="fresh");plt.plot(x,[by_iteration[i][1] for i in x],label="reused");plt.xlabel("refinement index");plt.ylabel("current-block token updates");plt.title("F1 fresh/reused work (GSM8K-32)");plt.legend();plt.tight_layout();plt.savefig(OUT/"02_f1_fresh_reused_over_refinement.png",dpi=180);plt.close()

pairs=[788011776,723984512,747224248]
plt.figure(figsize=(7,4));plt.bar(methods,pairs);plt.ylabel("fresh expert-token pairs");plt.title("Measured route work, matched GSM8K-32");plt.tight_layout();plt.savefig(OUT/"03_fresh_expert_pairs_partial.png",dpi=180);plt.close()

proj=json.load(open("artifacts/main_table/f1_gsm8k32_ep_projection.json"))
for number,metric,label in [(4,"routed_moe_stage_ms_per_request","routed-MoE ms/request"),(5,"max_mean","mean max/mean")]:
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for axis,ep in zip(axes,("ep4","ep8")):
        axis.bar(methods,[proj[ep][m][metric] for m in methods]);axis.set_title(ep.upper());axis.set_ylabel(label)
    fig.suptitle("EP2-calibrated projection, matched GSM8K-32");fig.tight_layout();fig.savefig(OUT/f"{number:02d}_{metric}_partial.png",dpi=180);plt.close(fig)

plt.figure(figsize=(6,4));plt.scatter([31/32,31/32,29/32],pairs);[(plt.annotate(m,(q,p))) for m,q,p in zip(methods,[31/32,31/32,29/32],pairs)];plt.xlabel("GSM8K-32 accuracy");plt.ylabel("fresh expert-token pairs");plt.title("Quality/work gate");plt.tight_layout();plt.savefig(OUT/"07_quality_pairs_partial.png",dpi=180);plt.close()
plt.figure(figsize=(6,4));cost=[proj["ep8"][m]["routed_moe_stage_ms_per_request"] for m in methods];plt.scatter([31/32,31/32,29/32],cost);[(plt.annotate(m,(q,p))) for m,q,p in zip(methods,[31/32,31/32,29/32],cost)];plt.xlabel("GSM8K-32 accuracy");plt.ylabel("simulated EP8 routed-MoE ms/request");plt.title("Quality/EP8 gate");plt.tight_layout();plt.savefig(OUT/"08_quality_ep8_partial.png",dpi=180);plt.close()

drift=json.load(open("artifacts/main_table/f1_gsm8k32_drift.json"));state=drift["staleness"]["1"]
plt.figure(figsize=(7,4));names=["expert Jaccard","EP4 rank Jaccard","EP8 rank Jaccard","commit disagreement"]
values=[state["expert_jaccard_mean"],state["ep4_rank_jaccard_mean"],state["ep8_rank_jaccard_mean"],state["commit_disagreement"]]
plt.bar(names,values);plt.xticks(rotation=20);plt.ylim(0,1);plt.title("F1 age-1 drift diagnostics");plt.tight_layout();plt.savefig(OUT/"09_f1_age1_drift.png",dpi=180);plt.close()
