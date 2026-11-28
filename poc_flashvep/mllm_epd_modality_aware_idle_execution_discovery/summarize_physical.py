"""Reduce physical vision and D-side decode probes from owned H200 epochs."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median


BUDGETS_MS = (5, 10, 25, 50, 100, 250, 500, 750)


def quantile(values, level):
    sorted_values = sorted(values)
    index = level*(len(values)-1)
    lo,hi = math.floor(index),math.ceil(index)
    return sorted_values[lo]+(sorted_values[hi]-sorted_values[lo])*(index-lo)


def rows_from(epochs):
    for epoch in epochs:
        path = epoch/"raw"/"vision_activity.jsonl"
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    yield json.loads(line) | {"epoch":epoch.name}


def phase_gpu(row, include_copies=True):
    two = row["E_D"]
    input_copy = two["D"]["input_ipc"]["gpu_ms"] if include_copies else 0
    return_copy = two["E_join"]["return_ipc"]["gpu_ms"] if include_copies else 0
    return two["E_patch"]["patch"]["gpu_ms"] + max(
        two["E_kept"]["prefix"]["gpu_ms"],
        input_copy + two["D"]["prefix"]["gpu_ms"]
    ) + return_copy + two["E_join"]["tail"]["gpu_ms"]


def reduce(records):
    layouts = {}
    image_groups = defaultdict(list)
    window_groups = defaultdict(list)
    balance_groups = defaultdict(list)
    intervention_groups = defaultdict(list)
    for row in records:
        key = (row["epoch"],row["fixture"])
        if row["kind"]=="layout":
            layouts[key] = row["E"]
        elif row["kind"]=="image":
            image_groups[key].append(row)
        elif row["kind"]=="window":
            window_groups[(*key,row["cut"])].append(row)
        elif row["kind"]=="image_balance":
            balance_groups[key].append(row)
        elif row["kind"]=="decode_interference":
            intervention_groups[(*key,row["mode"])].append(row)

    images = []
    for key, rows in sorted(image_groups.items()):
        geom = layouts[key]
        times = [row["D_full"]["D"]["vision"]["gpu_ms"] for row in rows]
        wall = [row["D_full"]["elapsed_ms"] for row in rows]
        check = [row["D_full"]["E"]["check"] for row in rows]
        images.append({"epoch":key[0],"fixture":key[1],"images":len(geom["grid"]),
                       "patch_rows":geom["patch_rows"],"rounds":len(rows),
                       "E_native_gpu_ms":median(row["E_native"]["vision"]["gpu_ms"] for row in rows),
                       "E_native_RPC_ms":median(row["E_native"]["rpc_wall_ms"] for row in rows),
                       "D_gpu_ms":median(times),"D_end_to_end_RPC_ms":median(wall),
                       "D_wall_p95_ms":quantile(wall,.95),
                       "D_H2D_and_cast_ms":median(row["D_full"]["D"]["input_h2d"]["gpu_ms"] for row in rows),
                       "D_E_clone_ms":median(row["D_full"]["E"]["return_ipc"]["gpu_ms"] for row in rows),
                       "returned_bytes":rows[0]["D_full"]["sent_bytes"],
                       "bitwise_trials":sum(w["bitwise"] for w in check),
                       "max_abs":max(w["max_abs"] for w in check),
                       "max_relative_l2":max(w["relative_l2"] for w in check)})

    windows = []
    for key, rows in sorted(window_groups.items()):
        check = [r["E_D"]["E_join"]["check"] for r in rows]
        native = median(r["E_native"]["vision"]["gpu_ms"] for r in rows)
        ideal = median(phase_gpu(r) for r in rows)
        zero_copy = median(phase_gpu(r, include_copies=False) for r in rows)
        same = median(r["E_same_cut"]["rpc_wall_ms"] for r in rows)
        end = median(r["E_D"]["elapsed_ms"] for r in rows)
        windows.append({"epoch":key[0],"fixture":key[1],"cut":key[2],"rounds":len(rows),
                        "remote_rows":rows[0]["E_D"]["E_patch"]["rows_sent"],
                        "total_ipc_bytes":rows[0]["E_D"]["sent_bytes"]+rows[0]["E_D"]["received_bytes"],
                        "D_prefix_gpu_ms":median(r["E_D"]["D"]["prefix"]["gpu_ms"] for r in rows),
                        "E_native_gpu_ms":native,"ideal_overlap_from_measured_phases_ms":ideal,
                        "ideal_gain":1-ideal/native,
                        "zero_copy_overlap_ms":zero_copy,"zero_copy_gain":1-zero_copy/native,
                        "E_tail_gpu_ms":median(r["E_D"]["E_join"]["tail"]["gpu_ms"] for r in rows),
                        "E_same_cut_RPC_ms":same,"E_D_join_ACK_wall_ms":end,
                        "E_native_RPC_ms":median(r["E_native"]["rpc_wall_ms"] for r in rows),
                        "bitwise_trials":sum(x["bitwise"] for x in check),
                        "local_same_cut_bitwise":sum(r["E_same_cut"]["check"]["bitwise"] for r in rows),
                        "local_frontier_bitwise":sum(r["E_frontier"]["check"]["bitwise"] for r in rows),
                        "local_same_cut_summary_matches":sum(
                            r["E_same_cut"]["check"]==comparison for r,comparison in zip(rows,check,strict=True)),
                        "max_abs":max(x["max_abs"] for x in check),
                        "max_relative_l2":max(x["relative_l2"] for x in check)})

    balanced = []
    for key, rows in sorted(balance_groups.items()):
        ms = {"E_baseline_gpu_ms":median(r["E_native"]["vision"]["gpu_ms"] for r in rows),
              "E_baseline_RPC_ms":median(r["E_native"]["rpc_wall_ms"] for r in rows),
              "E_D_wall_ms":median(r["E_D"]["elapsed_ms"] for r in rows),
              "E_keep_ms":median(r["E_D"]["E"]["vision"]["gpu_ms"] for r in rows),
              "D_encode_ms":median(r["E_D"]["D"]["vision"]["gpu_ms"] for r in rows)}
        balanced.append({"epoch":key[0],"fixture":key[1],"rounds":len(rows),
                          "offloaded_images":len(rows[0]["E_D"]["remote_images"]),
                          "offloaded_visual_tokens":rows[0]["E_D"]["sent_bytes"]//7168,
                          "returned_bytes":rows[0]["E_D"]["sent_bytes"],
                          "bitwise_trials":sum(r["E_D"]["join"]["check"]["bitwise"] for r in rows),
                          "max_relative_l2":max(r["E_D"]["join"]["check"]["relative_l2"] for r in rows),
                          **ms})
    intervention = []
    for key, rows in sorted(intervention_groups.items()):
        baseline = intervention_groups.get((key[0],"image_0","clean"), [])
        agreement = {r["repeat"]:r["token_ids"] for r in baseline}
        gaps = [x for row in rows for x in row["itl_ms"]]
        arrivals = [1000*(r["vision_complete"]-r["decode_submit"]) for r in rows
                    if r["vision_submit"] is not None and r["vision_submit"]<=r["decode_submit"]]
        intervention.append({"epoch":key[0],"fixture":key[1],"mode":key[2],"rounds":len(rows),
                             "D_submit_to_first_ms":median(r["D_submit_first_ms"] for r in rows),
                             "D_submit_to_first_p95_ms":quantile([r["D_submit_first_ms"] for r in rows],.95),
                             "TPOT_p95_ms":quantile([r["tpot_ms"] for r in rows],.95),
                             "ITL_p99_ms":quantile(gaps,.99),"ITL_max_ms":max(gaps),
                             "absolute_decode_pass":sum(r["tpot_ms"]<=33 and max(r["itl_ms"])<=34 for r in rows),
                             "exact_vs_same_rep_clean":sum(r["token_ids"]==agreement.get(r["repeat"]) for r in rows),
                             "vision_RPC_overlaps_decode_stream":sum(
                                 r["vision_submit"] is not None and r["vision_submit"]<r["last_token"]
                                 and r["first_token"]<r["vision_complete"] for r in rows),
                             "D_submitted_before_vision_completion":sum(x>0 for x in arrivals),
                             "vision_RPC_ms":None if key[2]=="clean" else median(
                                 1000*(r["vision_complete"]-r["vision_submit"]) for r in rows)})
    return {"layout":[{"epoch":k[0],"fixture":k[1],**v} for k,v in sorted(layouts.items())],
            "image":images,"window":windows,"image_balance":balanced,"decode_interference":intervention,
            "budget":[{"ms":b,"image_fixtures_by_RPC_p95":sum(x["D_wall_p95_ms"]<=b for x in images),
                       "window_sets_by_D_prefix_p50_only":sum(x["D_prefix_gpu_ms"]<=b for x in windows)}
                      for b in BUDGETS_MS],
            "scope":"Actual worker tests. Ideal overlap uses observed individual phases and excludes control RPC."}


def plot(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (items:=data["image"]):
        fig,ax=plt.subplots(figsize=(8.4,4),layout="constrained")
        row=[x["patch_rows"] for x in items]
        ax.scatter(row,[x["D_gpu_ms"] for x in items],label="D Vision CUDA",s=42)
        ax.scatter(row,[x["D_wall_p95_ms"] for x in items],
                   label="D upload + vision + return IPC / ACK p95",s=42)
        ax.set(xscale="log",yscale="log",xlabel="Original vision patch rows",
               ylabel="Time (ms)",title="Native vision modules on the existing Decode H200")
        ax.legend(frameon=False)
        fig.savefig(out/"idle_physical_image_work.svg")
        plt.close(fig)
    if (items:=data["window"]):
        fig,ax=plt.subplots(figsize=(8.4,4),layout="constrained")
        ax.scatter([x["total_ipc_bytes"]/2**20 for x in items],
                   [x["ideal_gain"]*100 for x in items],label="Optimistic phase overlap",s=35)
        ax.scatter([x["total_ipc_bytes"]/2**20 for x in items],
                   [x["zero_copy_gain"]*100 for x in items],label="Zero-copy phase estimate",s=30,marker="^")
        ax.scatter([x["total_ipc_bytes"]/2**20 for x in items],
                   [100*(1-x["E_D_join_ACK_wall_ms"]/x["E_native_RPC_ms"]) for x in items],
                   label="Measured complete RPC and ACK critical path",s=35)
        ax.axhline(10,ls="--",lw=1,color="grey")
        ax.set(xscale="log",xlabel="Round-trip activation bytes (MiB)",ylabel="Gain over E-only (%)",
               title="Exact window partition before the first full-attention block")
        ax.legend(frameon=False)
        fig.savefig(out/"idle_window_partition_transfer.svg")
        plt.close(fig)
    if data["image"] and data["window"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        budget = data["budget"]
        seconds = [x["ms"] for x in budget]
        axes[0].step(seconds, [x["image_fixtures_by_RPC_p95"] for x in budget], where="post")
        axes[0].set(title="Complete D image round, sampled p95", ylabel="Fixtures fitting budget")
        axes[1].step(seconds, [x["window_sets_by_D_prefix_p50_only"] for x in budget], where="post")
        axes[1].set(title="D window compute phase, median", ylabel="Cuts fitting compute budget")
        for axis in axes:
            axis.set(xscale="log", xlabel="Budget (ms)", ylim=(0,None))
            axis.set_xticks([5,25,100,250,750], labels=[5,25,100,250,750])
        fig.suptitle("Measured D work budgets with different completion boundaries")
        fig.savefig(out/"idle_physical_work_budget.svg")
        plt.close(fig)
    if (items:=data["decode_interference"]):
        labels=[f'{x["fixture"]} · {x["mode"]}' for x in items]
        fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
        ax.plot(labels,[x["ITL_p99_ms"] for x in items],"o-",label="ITL p99")
        ax.plot(labels,[x["ITL_max_ms"] for x in items],"s-",label="ITL max")
        ax.plot(labels,[x["TPOT_p95_ms"] for x in items],"^-",label="TPOT p95")
        ax.axhline(34,color="grey",ls="--")
        ax.tick_params(axis="x",rotation=22)
        ax.set(ylabel="Time (ms)",title="Native decode under controlled D-side vision arrival")
        ax.legend(frameon=False)
        fig.savefig(out/"idle_physical_decode_interference.svg")
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",action="append",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    data=reduce(rows_from(args.epoch))
    (args.out/"vision_physical_summary.json").write_text(json.dumps(data,indent=2)+"\n")
    figures=args.out/"figures"
    figures.mkdir(exist_ok=True)
    plot(data,figures)
    print({name:len(data[name]) for name in ("layout","image","window","image_balance","decode_interference")})


if __name__=="__main__":
    main()
