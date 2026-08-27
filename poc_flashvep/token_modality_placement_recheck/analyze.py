#!/usr/bin/env python3
"""Token-level Vision/Text placement recheck using only real-image traces.

The paired text-control routes are deliberately not loaded into any primary
fit/evaluation. This script is offline-only and never initializes CUDA/vLLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from poc_flashvep.modality_placement_saturation_tradeoff.analyze import (  # noqa: E402
    EP,
    EXPERTS,
    IMAGE_TOKEN_ID,
    LAYERS,
    LOCAL_EXPERTS,
    TOPK,
    Trace,
    concat_routes,
    load_traces,
    make_placement,
    saturation_metrics,
    source_positions,
)

SEED = 20260827
POLICIES = ["P0", "P_load", "P_V_load", "P_T_load", "P_V_sat", "P_T_sat", "P_joint_0.5", "P_joint_0.75"]


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def source_routes(trace: Trace, layer: int, source: str) -> np.ndarray:
    return trace.routes[source_positions(trace, source), layer, :]


def combined(traces: list[Trace], layer: int, source: str) -> np.ndarray:
    chunks = [source_routes(t, layer, source) for t in traces]
    chunks = [x for x in chunks if len(x)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 8), dtype=np.int64)


def current_rows(visions: list[Trace], out: Path) -> pd.DataFrame:
    rows = []
    p = np.arange(EXPERTS, dtype=np.int64) // LOCAL_EXPERTS
    for source in ("vision", "text", "all"):
        for scope in ("global", "layer", "request_layer"):
            if scope == "global":
                groups = [("global", visions, None)]
            elif scope == "layer":
                groups = [(str(layer), visions, layer) for layer in range(LAYERS)]
            else:
                groups = [(f"{t.request_id}:L{layer}", [t], layer) for t in visions for layer in range(LAYERS)]
            for key, traces, layer in groups:
                if layer is None:
                    chunks = [combined(traces, l, source) for l in range(LAYERS)]
                    chunks = [x for x in chunks if len(x)]
                    route = np.concatenate(chunks, axis=0)
                else:
                    route = combined(traces, layer, source)
                m = saturation_metrics(route, p)
                rows.append({"scope": scope, "key": key, "source": source, **{k: v for k, v in m.items() if k != "rank_loads"}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "current_saturation.csv", index=False)
    return df


def fit_routes(traces: list[Trace], layer: int, fit_source: str, policy: str) -> tuple[np.ndarray, str]:
    route = combined(traces, layer, fit_source)
    return make_placement(route, policy)


def evaluate(visions: list[Trace], out: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    placements: dict[str, dict] = {"layers": {}}
    for layer in range(LAYERS):
        placements["layers"][str(layer)] = {}
        for policy in POLICIES:
            if policy == "P0":
                fit_source = "all"
            elif policy == "P_V_load" or policy == "P_V_sat":
                fit_source = "vision"
            elif policy == "P_T_load" or policy == "P_T_sat":
                fit_source = "text"
            else:
                fit_source = "all"
            p, optimizer = fit_routes(visions, layer, fit_source, policy)
            if np.bincount(p, minlength=EP).tolist() != [LOCAL_EXPERTS] * EP:
                raise AssertionError(f"capacity violation {policy} layer {layer}")
            placements["layers"][str(layer)][policy] = {"fit_source": fit_source, "optimizer": optimizer, "expert_to_rank": p.tolist()}
            for eval_source in ("all", "vision", "text"):
                route = combined(visions, layer, eval_source)
                m = saturation_metrics(route, p)
                rows.append({"layer": layer, "placement": policy, "fit_source": fit_source, "eval_source": eval_source, "optimizer": optimizer, **{k: v for k, v in m.items() if k != "rank_loads"}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "placement_frontier.csv", index=False)
    write_json(out / "placement_assignments.json", placements)
    return df, placements


def cross_table(frontier: pd.DataFrame, out: Path) -> pd.DataFrame:
    cols = ["rank_load_cv", "rank_load_max_mean", "mean_u", "p_u4", "p_u_ge3", "remote_volume_proxy"]
    rows = []
    for p, g in frontier.groupby("placement"):
        for src, h in g.groupby("eval_source"):
            row = {"placement": p, "eval_source": src, "fit_source": h.fit_source.iloc[0]}
            for c in cols:
                row[c] = float(h[c].mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "modality_cross_eval.csv", index=False)
    return df


def heldout(visions: list[Trace], out: Path) -> pd.DataFrame:
    rows = []
    folds = [("A_first12", visions[:12], visions[12:]), ("B_last12", visions[12:], visions[:12])]
    for fold, cal, ev in folds:
        for policy in ("P0", "P_V_load", "P_T_load", "P_joint_0.5", "P_joint_0.75"):
            fit_source = "all" if policy in ("P0", "P_joint_0.5", "P_joint_0.75") else ("vision" if policy == "P_V_load" else "text")
            for layer in range(LAYERS):
                p, optimizer = fit_routes(cal, layer, fit_source, policy)
                for eval_source in ("all", "vision", "text"):
                    m = saturation_metrics(combined(ev, layer, eval_source), p)
                    rows.append({"fold": fold, "policy": policy, "fit_source": fit_source, "eval_source": eval_source, "layer": layer, "optimizer": optimizer, **{k: v for k, v in m.items() if k != "rank_loads"}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "heldout_transfer.csv", index=False)
    return df


def figures(current: pd.DataFrame, frontier: pd.DataFrame, cross: pd.DataFrame, transfer: pd.DataFrame, out: Path) -> list[str]:
    fd = out / "figures"; fd.mkdir(parents=True, exist_ok=True); paths = []
    # Figure 1: token-level current saturation distribution.
    fig, ax = plt.subplots(figsize=(7, 4))
    for source, color in (("vision", "#4c78a8"), ("text", "#f58518")):
        q = current[(current.scope == "request_layer") & (current.source == source)]
        # request-layer rows are aggregate metrics, so the histogram is over the
        # integer u proxy's mean rather than raw token counts.
        ax.hist(q.mean_u / EP, bins=20, alpha=.6, label=source, color=color)
    ax.set_xlabel("request-layer mean unique-rank saturation u/4"); ax.set_ylabel("request-layer count"); ax.legend(); ax.set_title("Token-level Vision/Text saturation under P0")
    fig.tight_layout(); p=fd/"plot1_token_level_saturation.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    # Figure 2: cross-evaluation heatmap.
    names = [x for x in POLICIES if x in set(cross.placement)]
    mat=[]
    for name in names:
        row=[]
        for src in ("vision","text"):
            q=cross[(cross.placement==name)&(cross.eval_source==src)].iloc[0]
            row += [q.rank_load_max_mean, q.mean_u/EP]
        mat.append(row)
    fig, ax=plt.subplots(figsize=(8,5)); im=ax.imshow(np.asarray(mat),aspect="auto",cmap="magma_r"); ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8); ax.set_xticks(range(4)); ax.set_xticklabels(["V max/mean","V S","T max/mean","T S"]); fig.colorbar(im,ax=ax,label="lower is better"); ax.set_title("Token-level modality placement cross-evaluation"); fig.tight_layout(); p=fd/"plot2_token_modality_cross_eval.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    # Figure 3: fixed fold held-out max/mean, modality split.
    q=transfer[transfer.eval_source.isin(["vision","text"])].groupby(["fold","policy","eval_source"],as_index=False).rank_load_max_mean.mean()
    fig, ax=plt.subplots(figsize=(11,4.5)); x=np.arange(len(q)); ax.bar(x,q.rank_load_max_mean); ax.set_xticks(x); ax.set_xticklabels([f"{a}\n{b}\n{c}" for a,b,c in zip(q.fold,q.policy,q.eval_source)],rotation=70,fontsize=7); ax.set_ylabel("held-out max/mean rank load"); ax.set_title("Calibration → held-out token-level transfer"); fig.tight_layout(); p=fd/"plot3_heldout_transfer.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    return paths


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/workload_manifest.json")); ap.add_argument("--route-root",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    visions,text_controls,manifest=load_traces(args.manifest,args.route_root)
    current=current_rows(visions,args.output); frontier,assignments=evaluate(visions,args.output); cross=cross_table(frontier,args.output); transfer=heldout(visions,args.output); fig_paths=figures(current,frontier,cross,transfer,args.output)
    shutil.copy2(args.manifest,args.output/"source_workload_manifest.json")
    write_json(args.output/"analysis_policy.json",{"primary_traces":"24 real-image vision traces only; paired text-control routes excluded from all primary fit/eval","image_token_id":IMAGE_TOKEN_ID,"ep":EP,"experts":EXPERTS,"layers":LAYERS,"top_k":TOPK,"capacity":"32 experts per rank","policies":POLICIES,"joint_lambdas":[0.5,0.75],"placement_fit":"per layer; P_load all real-image tokens, P_V/P_T token sources, P_joint all real-image tokens","optimizer":"deterministic heuristic; no placement called oracle","folds":"first 12 -> latter 12 and reverse","gpu_execution":False})
    def agg(policy, source):
        q=cross[(cross.placement==policy)&(cross.eval_source==source)].iloc[0]; return {k:float(q[k]) for k in ["rank_load_cv","rank_load_max_mean","mean_u","p_u4","p_u_ge3"]}
    p0v,p0t=agg("P0","vision"),agg("P0","text")
    pv,pt=agg("P_V_load","vision"),agg("P_T_load","text")
    pvt,ptv=agg("P_T_load","vision"),agg("P_V_load","text")
    pjointv,pjointt=agg("P_joint_0.5","vision"),agg("P_joint_0.5","text")
    # Explicit Hamming statistics (kept separate from the summary expression).
    ham=[]
    for l in range(LAYERS):
        a=assignments["layers"][str(l)]["P_V_load"]["expert_to_rank"]; b=assignments["layers"][str(l)]["P_T_load"]["expert_to_rank"]; ham.append(int(sum(x!=y for x,y in zip(a,b))))
    ht=transfer[(transfer.policy.isin(["P0","P_V_load","P_T_load","P_joint_0.5"]))&(transfer.eval_source.isin(["vision","text"]))].groupby(["fold","policy","eval_source"],as_index=False).mean(numeric_only=True)
    summary={"gpu_execution":False,"primary_trace_count":len(visions),"paired_text_controls_loaded_for_primary":False,"assignment_invariant":"vision+text token masks partition every real-image request; routes unchanged","current":{"vision":p0v,"text":p0t},"placement":{"P_V_load_on_Vision":pv,"P_T_load_on_Text":pt,"P_T_load_on_Vision":pvt,"P_V_load_on_Text":ptv,"P_joint_0.5_on_Vision":pjointv,"P_joint_0.5_on_Text":pjointt},"P_V_load_vs_P_T_load_hamming":{"mean":float(np.mean(ham)),"median":float(np.median(ham)),"min":int(min(ham)),"max":int(max(ham))},"heldout_summary":ht.to_dict(orient="records"),"figures":fig_paths}
    write_json(args.output/"summary.json",summary); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
