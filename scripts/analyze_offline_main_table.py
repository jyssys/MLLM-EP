#!/usr/bin/env python3
"""Build common quality/work and routed-MoE projection tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from scripts.analyze_h1_straggler_deepdive import BatchComputeModel
from virtual_ep.comm_model import CommunicationScenario


HIDDEN_BYTES = 2048 * 2
LABELS = {1: "LOCAL / NO EP COMM", 4: "SIMULATED-EP4-EP2-CALIBRATED",
          8: "SIMULATED-EP8-EP2-CALIBRATED"}


def jsonl_directory(directory: Path) -> dict[int, dict]:
    rows = {}
    for path in sorted(directory.glob("generations_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line); rows[int(row["sample_id"])] = row
    return rows


def baseline_rows(path: Path, limit: int = 128) -> dict[int, dict]:
    return {int(row["sample_id"]): row for row in map(json.loads, path.open())
            if int(row["sample_id"]) < limit}


def paired_quality(rows: dict[int, dict], baseline: dict[int, dict], field="correct") -> dict:
    ids = sorted(set(rows) & set(baseline))
    correct = lambda row: bool(row.get(field, False))
    lost = sum(correct(baseline[i]) and not correct(rows[i]) for i in ids)
    fixed = sum(not correct(baseline[i]) and correct(rows[i]) for i in ids)
    discordant = lost + fixed
    return {
        "n": len(ids), "correct": sum(correct(rows[i]) for i in ids),
        "accuracy": np.mean([correct(rows[i]) for i in ids]) if ids else None,
        "baseline_correct": sum(correct(baseline[i]) for i in ids),
        "baseline_correct_to_wrong": lost, "baseline_wrong_to_correct": fixed,
        "accuracy_delta_pp": (100 * (fixed - lost) / len(ids) if ids else None),
        "paired_exact_p": float(binomtest(lost, discordant, .5).pvalue) if discordant else 1.0,
    }


def work_summary(rows: dict[int, dict], trace_dir: Path | None = None,
                 trace_format: str | None = None) -> dict:
    values = list(rows.values())
    result = {
        "nfe_mean": float(np.mean([r["nfe"] for r in values])) if values else None,
        "termination": dict((name, sum((r.get("termination") or r.get("termination_reason")) == name
                                       for r in values))
                            for name in sorted({r.get("termination") or r.get("termination_reason")
                                                for r in values})),
        "w_active": sum(r.get("active_updates", 0) for r in values) or None,
    }
    if trace_dir is None: return result
    pairs = unique = fresh = reused = invocations = 0
    for path in trace_dir.glob("request_*.npz"):
        with np.load(path, allow_pickle=False) as z:
            if trace_format == "selective":
                hist = z["hist_s1"].reshape(-1, 256)
                fresh += int((~(z["masked"] & ~z["active"])).sum())
                reused += int((z["masked"] & ~z["active"]).sum())
            else:
                hist = z["hist"].reshape(-1, 256)
                fresh += int(z["fresh"].sum()); reused += int(z["reused"].sum())
            pairs += int(hist.sum()); unique += int(np.count_nonzero(hist)); invocations += len(hist)
    result.update({"fresh_expert_token_pairs": pairs,
                   "unique_activated_experts_per_forward": unique / max(invocations, 1),
                   "fresh_token_updates": fresh, "reused_token_updates": reused,
                   "avg_k": 8.0})
    return result


def load_method_shapes(directory: Path, trace_format: str, ep: int):
    histograms=[]; unique=[]; fanout=[]; requests=[]
    for path in sorted((directory / "traces").glob("request_*.npz")):
        rid = int(path.stem.split("_")[-1])
        with np.load(path, allow_pickle=False) as z:
            if trace_format == "selective":
                h=z["hist_s1"].reshape(-1,256)
                if ep == 1 and "u_s1_ep1" not in z:
                    u=np.zeros((len(h),1,1));f=np.zeros(len(h))
                else:
                    u=z[f"u_s1_ep{ep}"].reshape(-1,ep,ep)
                    f=z[f"fanout_mean_s1_ep{ep}"].reshape(-1)
            else:
                h=z["hist"].reshape(-1,256); u=z[f"u_ep{ep}"].reshape(-1,ep,ep)
                f=z[f"fanout_mean_ep{ep}"].reshape(-1)
        histograms.append(h); unique.append(u); fanout.append(f)
        requests.extend([rid]*len(h))
    return np.concatenate(histograms),np.concatenate(unique),np.concatenate(fanout),np.asarray(requests)


def load_vanilla_shapes(path: Path, ep: int):
    with np.load(path, allow_pickle=False) as z:
        hist=z["expert_counts_by_class"].sum(axis=1).astype(np.float64)
        if ep == 1:
            u=np.zeros((len(hist),1,1),dtype=np.float64);fanout=np.zeros(len(hist))
        else:
            u=z[f"unique_matrix_ep{ep}"].astype(np.float64)
            fanout=z[f"mean_fanout_ep{ep}"].astype(np.float64)
        request=z["request_id"].astype(np.int16)
    return hist,u,fanout,request


def project_shapes(hist, unique, fanout, request, ep, model, comm):
    width=256//ep
    vectors=np.column_stack([model.predict_batch(hist[:,r*width:(r+1)*width]) for r in range(ep)])
    dispatch=[];combine=[];remote=[];remote_fanout=[]
    for matrix in unique:
        rem=matrix.copy();np.fill_diagonal(rem,0)
        outgoing=rem.sum(axis=1)*HIDDEN_BYTES;incoming=rem.sum(axis=0)*HIDDEN_BYTES
        if ep==1: d=c=0.0
        else:
            d=comm.dispatch.latency(outgoing,incoming);c=comm.combine.latency(incoming,outgoing)
        dispatch.append(d);combine.append(c);remote.append(float(outgoing.sum()))
        remote_fanout.append(float(np.count_nonzero(rem,axis=1).mean()))
    dispatch=np.asarray(dispatch);combine=np.asarray(combine);expert=vectors.max(axis=1)
    stage=dispatch+expert+combine;mean=vectors.mean(axis=1);ordered=np.sort(vectors,axis=1)
    second=ordered[:,-2] if ep>1 else expert
    per_request=defaultdict(float)
    for rid,value in zip(request,stage):per_request[int(rid)]+=float(value)
    return {
        "requests":len(set(request.tolist())),"invocations":len(hist),
        "routed_moe_stage_ms_per_request":float(stage.sum()/len(set(request.tolist()))),
        "dispatch_ms_per_request":float(dispatch.sum()/len(set(request.tolist()))),
        "expert_ms_per_request":float(expert.sum()/len(set(request.tolist()))),
        "combine_ms_per_request":float(combine.sum()/len(set(request.tolist()))),
        "max_mean":float(np.mean(expert/np.maximum(mean,1e-12))),
        "max_second":float(np.mean(expert/np.maximum(second,1e-12))),
        "cv":float(np.mean(vectors.std(axis=1)/np.maximum(mean,1e-12))),
        "wait_fraction":float(np.mean((ep*expert-vectors.sum(axis=1))/np.maximum(ep*expert,1e-12))),
        "remote_logical_bytes_per_request":float(np.sum(remote)/len(set(request.tolist()))),
        "fanout":float(np.mean(remote_fanout)),"label":LABELS[ep],
        "request_stage_ms_p50":float(np.median(list(per_request.values()))),
        "request_stage_ms_p95":float(np.percentile(list(per_request.values()),95)),
    }


def markdown_table(rows, columns):
    out=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for row in rows: out.append("| "+" | ".join(str(row.get(c,"N/A")) for c in columns)+" |")
    return "\n".join(out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--f0",type=Path,required=True);parser.add_argument("--f1",type=Path,required=True)
    parser.add_argument("--team",type=Path,required=True);parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--discovery",type=Path,required=True);parser.add_argument("--comm",type=Path,required=True)
    parser.add_argument("--ep1-model",type=Path,required=True);parser.add_argument("--ep4-model",type=Path,required=True)
    parser.add_argument("--ep8-model",type=Path,required=True);parser.add_argument("--reports",type=Path,default=Path("reports"))
    args=parser.parse_args();args.reports.mkdir(exist_ok=True);figdir=args.reports/"figures/offline_main_table";figdir.mkdir(parents=True,exist_ok=True)
    base=baseline_rows(args.baseline); methods={
        "Vanilla":base,"TEAM-PORT":jsonl_directory(args.team),
        "OURS-F0":jsonl_directory(args.f0),"OURS-F1":jsonl_directory(args.f1)}
    quality={name:paired_quality(rows,base) for name,rows in methods.items()}
    work={"Vanilla":{"nfe_mean":float(np.mean([r["nfe"] for r in base.values()])),"avg_k":8.0,
                     "termination":dict((x,sum(r["termination_reason"]==x for r in base.values())) for x in {r["termination_reason"] for r in base.values()})},
          "TEAM-PORT":work_summary(methods["TEAM-PORT"],args.team/"traces","team"),
          "OURS-F0":work_summary(methods["OURS-F0"],args.f0/"traces","selective"),
          "OURS-F1":work_summary(methods["OURS-F1"],args.f1/"traces","selective")}
    with np.load(args.discovery, allow_pickle=False) as z:
        vanilla_hist=z["expert_counts_by_class"].sum(axis=1)
        first_layer=int(z["layer_id"][0]); first=z["layer_id"]==first_layer
        work["Vanilla"].update({
            "fresh_expert_token_pairs":int(vanilla_hist.sum()),
            "unique_activated_experts_per_forward":float(np.count_nonzero(vanilla_hist,axis=1).mean()),
            "w_active":int(z["masked_current_block"][first].sum()),
        })
    models={1:BatchComputeModel(args.ep1_model),4:BatchComputeModel(args.ep4_model),8:BatchComputeModel(args.ep8_model)}
    comm=CommunicationScenario.load(args.comm,"ep2_calibrated_base"); projection={}
    for ep in (1,4,8):
        projection[f"ep{ep}"]={}
        for name in methods:
            if name=="Vanilla": shapes=load_vanilla_shapes(args.discovery,ep)
            elif name=="TEAM-PORT": shapes=load_method_shapes(args.team,"team",ep)
            else: shapes=load_method_shapes(args.f0 if name=="OURS-F0" else args.f1,"selective",ep)
            projection[f"ep{ep}"][name]=project_shapes(*shapes,ep,models[ep],comm)
        vanilla=projection[f"ep{ep}"]["Vanilla"]["routed_moe_stage_ms_per_request"]
        for value in projection[f"ep{ep}"].values():value["speedup_vs_same_ep_vanilla"]=vanilla/value["routed_moe_stage_ms_per_request"]
        projection[f"ep{ep}"]["REFLEX"]={"label":LABELS[ep],"status":"N/A — official code unavailable (user-directed)"}
    summary={"quality":quality,"work":work,"projection":projection,
             "reflex":{"status":"N/A","reason":"official code unavailable; user directed no reimplementation"}}
    (args.reports/"offline_main_table_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    qa=[]
    for name in ("Vanilla","TEAM-PORT","REFLEX","OURS-F0","OURS-F1"):
        if name=="REFLEX": qa.append({"Method":name,"GSM8K":"N/A","NFE":"N/A","AvgK":"N/A","Fresh pairs":"N/A","W_active":"N/A","Termination":"N/A"});continue
        q=quality[name];w=work[name];qa.append({"Method":name,"GSM8K":f"{q['correct']}/{q['n']} ({100*q['accuracy']:.2f}%)","NFE":f"{w['nfe_mean']:.2f}","AvgK":w.get("avg_k",8),"Fresh pairs":w.get("fresh_expert_token_pairs","N/A"),"W_active":w.get("w_active","N/A"),"Termination":w["termination"]})
    (args.reports/"offline_main_table_quality_work.md").write_text("# Offline main table: quality and work\n\nLabels: quality is `MEASURED_QUALITY`; work counters are measured from actual rollouts. REFLEX is intentionally N/A.\n\n"+markdown_table(qa,["Method","GSM8K","NFE","AvgK","Fresh pairs","W_active","Termination"])+"\n")
    ep_rows=[]
    for ep in (1,4,8):
        for name,value in projection[f"ep{ep}"].items():
            if "status" in value: ep_rows.append({"Method":name,"EP":ep,"Stage":"N/A","Speedup":"N/A","Dispatch":"N/A","Expert":"N/A","Combine":"N/A","max/mean":"N/A","CV":"N/A","Remote bytes":"N/A","Label":value["label"]});continue
            ep_rows.append({"Method":name,"EP":ep,"Stage":f"{value['routed_moe_stage_ms_per_request']:.3f}","Speedup":f"{value['speedup_vs_same_ep_vanilla']:.3f}x","Dispatch":f"{value['dispatch_ms_per_request']:.3f}","Expert":f"{value['expert_ms_per_request']:.3f}","Combine":f"{value['combine_ms_per_request']:.3f}","max/mean":f"{value['max_mean']:.3f}","CV":f"{value['cv']:.3f}","Remote bytes":f"{value['remote_logical_bytes_per_request']:.0f}","Label":value["label"]})
    (args.reports/"offline_main_table_ep_projection.md").write_text("# Offline main table: routed-MoE projection\n\nThese are routed-MoE stage projections, **not E2E latency**. EP1 is local grouped expert replay with zero EP communication. EP4/EP8 use the held-out-validated EP2 calibration.\n\n"+markdown_table(ep_rows,["Method","EP","Stage","Speedup","Dispatch","Expert","Combine","max/mean","CV","Remote bytes","Label"])+"\n")
    # Required overview figures; method-specific diagnostic figures are made by
    # the F1 report generator to avoid duplicating heavy trace traversal.
    names=["Vanilla","TEAM-PORT","OURS-F0","OURS-F1"]
    for ep in (1,4,8):
        vals=[projection[f"ep{ep}"][n]["routed_moe_stage_ms_per_request"] for n in names]
        plt.figure(figsize=(7,4));plt.bar(names,vals);plt.ylabel("routed-MoE ms/request");plt.title(f"EP{ep} common projection");plt.xticks(rotation=20);plt.tight_layout();plt.savefig(figdir/f"ep{ep}_stage.png",dpi=160);plt.close()
    for ep in (4,8):
        vals=[projection[f"ep{ep}"][n]["max_mean"] for n in names]
        plt.figure(figsize=(7,4));plt.bar(names,vals);plt.ylabel("mean max/mean rank time");plt.title(f"EP{ep} imbalance");plt.xticks(rotation=20);plt.tight_layout();plt.savefig(figdir/f"ep{ep}_max_mean.png",dpi=160);plt.close()


if __name__=="__main__":main()
