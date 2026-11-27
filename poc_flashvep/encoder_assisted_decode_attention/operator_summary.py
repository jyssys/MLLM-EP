"""Summarize the physical operator matrix and reproduce the partition and timing plots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def read_rows(inputs):
    for path in inputs:
        with path.open() as stream:
            lines = list(map(json.loads,stream))
        if lines[-1]["type"] != "complete":
            raise ValueError(f"Cannot summarize incomplete operator log {path}")
        config = lines[0]
        for case in lines[1:-1]:
            for outcome in case["outcomes"]:
                values = outcome["timings"]
                base = median(record["cuda_us"] for record in values["local_best_graph"])
                measurement = {name:median(row["cuda_us"] for row in measures) for name,measures in values.items()}
                traces = outcome["instrumented_eager_us"]
                breakdown = {key: median(row[key] for row in traces) for key in traces[0]}
                yield {
                    "run":path.stem,"layout":config["layout"],"transport":config["graph_transport"],
                    "workload":case["shape_type"],"batch":len(case["context"]),
                    "max_context":max(case["context"]),"total_rows":sum(case["context"]),
                    "q_heads":case["q_heads"],"kv_heads":case["kv_heads"],"head_dim":case["head_dim"],
                    "mode":outcome["placement"],"requested_fraction":outcome["requested_remote_fraction"],
                    "actual_fraction":sum(outcome["remote_lengths"])/sum(case["context"]),
                    "local_graph_us":base,"best_local_num_splits":case["chosen_num_splits"],
                    "local_eager_us":measurement["local_eager"],
                    "split_eager_us":measurement["split_eager"],
                    "split_graph_us":measurement.get("split_graph"),
                    "graph_improvement":1-measurement["split_graph"]/base if "split_graph" in measurement else None,
                    "eager_l2":outcome["correctness"]["relative_l2"],
                    "lse_error":outcome["correctness"].get("lse_max_abs"),
                    "graph_l2":(outcome["graph_correctness"] or {}).get("relative_l2"),
                    "graph_err":outcome.get("graph_error"),
                    "graph_replay_l2":(outcome["graph_changed_query_correctness"] or {}).get("relative_l2"),
                    "link_bytes":sum(outcome["recurrent_link_bytes"].values()),
                    "local_kv_bytes":case["full_physical_kv_bytes"],
                    "balanced_d_bytes":outcome["destination_bytes"]["decode"],
                    "balanced_e_bytes":outcome["destination_bytes"]["helper"],
                    "full_setup_ms":case["full_placement_ms"],
                    "split_setup_ms":outcome["partition_setup_ms"],
                    "nvlink":json.dumps(outcome["nvlink_data"],ensure_ascii=False),
                    **{f"eager_{key}_us": value for key,value in breakdown.items()},
                }


def figures(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = defaultdict(list)
    for row in rows:
        if row["mode"] == "generic_prefix" and row["split_graph_us"] is not None and row["layout"] == "paged":
            selected[(row["run"],row["max_context"],row["batch"])].append(row)
    if selected:
        best_run = max({key[0] for key in selected},key=lambda run:sum(key[0]==run for key in selected))
        curves = [(ctx,b,sorted(data,key=lambda row:row["actual_fraction"])) for (run,ctx,b),data in selected.items()
                  if run==best_run and len(data)>=3]
        highest = sorted(curves,key=lambda case:case[0]*case[1],reverse=True)[:5]
        if highest:
            fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
            for ctx,b,data in reversed(highest):
                ax.plot([0,*[100*row["actual_fraction"] for row in data]],
                        [data[0]["local_graph_us"],*[row["split_graph_us"] for row in data]],"o-",
                        label=f"B{b} × {ctx:,} rows")
            ax.set_xlabel("KV rows on helper (%)")
            ax.set_ylabel("Decode attention round-trip (µs)")
            ax.legend(frameon=False)
            ax.set_title("Physical peer transfer, FA3 and merge")
            fig.savefig(out/"partition_attention.png",dpi=170)
            plt.close(fig)
    biggest=max(rows,key=lambda row:row["total_rows"]*(-1000*abs(row["actual_fraction"]-.5)+500))
    order=("query_copy","remote_attention","result_copy","local_attention","wait_for_remote","merge")
    vals=[biggest[f"eager_{name}_us"] for name in order]
    fig,ax=plt.subplots(figsize=(7,3.8),layout="constrained")
    ax.barh(range(len(order)),vals,color=["#5182af"]*3+["#b9644a"]*3)
    ax.set_yticks(range(len(order)),[n.replace("_"," ") for n in order])
    ax.invert_yaxis()
    ax.set_xlabel("Median CUDA event enclosure (µs)")
    ax.set_title(f"Eager timing · B{biggest['batch']} × {biggest['max_context']:,} rows")
    fig.savefig(out/"communication_enclosures.png",dpi=170)
    plt.close(fig)


def interference_payload(source):
    payload = list(map(json.loads,source.read_text().splitlines()))
    if payload[-1]["type"] != "complete":
        raise ValueError("Finish the encoder intervention before plotting it")
    return payload[0],payload[1:-1]


def interference_rows(source):
    _,raw = interference_payload(source)
    for row in raw:
        by_mode = defaultdict(list)
        for item in row["measurements"]:
            by_mode[item["mode"]].append(item)
        observed = [item["physical_E_event_overlap"] for item in by_mode["observed_assist"]]
        yield {
            "request":row["request_id"],"shape_source":row["shape_source"],"kv_rows":tuple(row["kv_rows"]),
            "rows":row["visual_rows"],"images":row["image_count"],
            **{f"{mode}_D_us":median(item["D"]["cuda_us"] for item in items)
               for mode,items in by_mode.items() if items and items[0]["D"] is not None},
            **{f"{mode}_E_ms":median(item["encoder_gpu_ms"] for item in items)
               for mode,items in by_mode.items() if items and items[0]["encoder_gpu_ms"] is not None},
            "encoder_first_D_queue_ms":median(item["D_queue_event_ms"] for item in by_mode["encoder_first"])
            if by_mode["encoder_first"] else None,
            "E_intersection_observed":sum(item["intersect_ms"]>0 for item in observed),
            "E_intersection_attempted":len(observed),
        }


def interference_figure(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scoped = defaultdict(list)
    for row in rows:
        scoped[(row["shape_source"],row["kv_rows"])].append(row)
    fig, axs = plt.subplots(len(scoped),2,figsize=(9,3.8*len(scoped)),layout="constrained",squeeze=False)
    for ((_,context),data),axes in zip(scoped.items(),axs,strict=True):
        data.sort(key=lambda row:row["rows"])
        x = [r["rows"] for r in data]
        for key,label in (("isolated_local_D_us","Local"),("local_with_encode_D_us","Local + Encode"),
                          ("isolated_assist_D_us","EADA"),("assist_with_encode_D_us","EADA + Encode")):
            axes[0].plot(x,[r[key] for r in data],"o-",label=label)
        for key,label in (("encoder_only_E_ms","Encode alone"),("local_with_encode_E_ms","Encode + Local"),
                          ("assist_with_encode_E_ms","Encode + EADA")):
            axes[1].plot(x,[r[key] for r in data],"o-",label=label)
        axes[0].set_ylabel("D latency (µs per attention)")
        axes[1].set_ylabel("E encoder enclosure (ms)")
        axes[0].set_title(f"B{len(context)} × max {max(context):,} KV rows")
        axes[1].set_title("Original vision encoder")
        for ax in axes:
            ax.set_xlabel("Actual visual prompt rows")
            ax.legend(frameon=False)
    fig.savefig(out/"encoder_interference.png",dpi=170)
    plt.close(fig)


def compare_priorities(canonical, equal):
    a_config,a_cases = interference_payload(canonical)
    b_config,b_cases = interference_payload(equal)
    a_priority = a_config.pop("encoder_priority")
    b_priority = b_config.pop("encoder_priority")
    if (a_priority,b_priority) != (-1,0) or a_config != b_config:
        raise ValueError("Priority sensitivity requires the same devices, checkpoint and invocation contract")
    cases = ("shape_source","kv_rows","request_id","helper_fraction","visual_rows","image_count")
    observations = ("replicate","mode")
    for a,b in zip(a_cases,b_cases,strict=True):
        if any(a[field] != b[field] for field in cases):
            raise ValueError("The paired encoder intervention must use the same workload")
        order = lambda case: [tuple(item[key] for key in observations) for item in case["measurements"]]
        if [entry for entry in order(a) if entry[1] != "observed_assist"] != [
                entry for entry in order(b) if entry[1] != "observed_assist"]:
            raise ValueError("The paired encoder intervention must preserve randomized order")
    for a,b in zip(interference_rows(canonical),interference_rows(equal),strict=True):
        row = {key:a[key] for key in ("request","shape_source","kv_rows","rows","images")}
        for label,measurement in (("priority_minus_1",a),("priority_0",b)):
            row.update({f"{label}_{key}":value for key,value in measurement.items() if key not in row})
            for device,unit in (("D","us"),("E","ms")):
                treatment = measurement[f"assist_with_encode_{device}_{unit}"]
                baseline = measurement[f"local_with_encode_{device}_{unit}"]
                row[f"{label}_{device}_delta_pct"] = 100*(treatment/baseline-1)
            row[f"{label}_D_isolated_delta_pct"] = 100*(
                measurement["isolated_assist_D_us"]/measurement["isolated_local_D_us"]-1)
        for device in ("D","E"):
            row[f"change_{device}_percentage_points"] = row[f"priority_0_{device}_delta_pct"]-row[f"priority_minus_1_{device}_delta_pct"]
        yield row


def priority_figure(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = defaultdict(list)
    for row in rows:
        shapes[row["kv_rows"]].append(row)
    fig,axes = plt.subplots(len(shapes),2,figsize=(9,3.8*len(shapes)),layout="constrained",squeeze=False)
    for (context,data),plots in zip(shapes.items(),axes,strict=True):
        data.sort(key=lambda row:row["rows"])
        x = [r["rows"] for r in data]
        for ax,device,title in zip(plots,("D","E"),("Attention","Vision encoder"),strict=True):
            ax.axhline(0,color="0.65",linewidth=1)
            for key,label in (("priority_minus_1","Encoder priority −1"),("priority_0","Equal priority 0")):
                ax.plot(x,[r[f"{key}_{device}_delta_pct"] for r in data],"o-",label=label)
            ax.set_title(f"{title} · B{len(context)} × max {max(context):,} KV rows")
            ax.set_xlabel("Actual visual prompt rows")
            ax.set_ylabel("EADA latency change vs concurrent Local (%)")
            ax.legend(frameon=False)
    fig.savefig(out/"encoder_priority_sensitivity.png",dpi=170)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w",newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--encoder-run",type=Path)
    parser.add_argument("--encoder-equal-priority-run",type=Path)
    parser.add_argument("runs",type=Path,nargs="+")
    args=parser.parse_args()
    if args.encoder_equal_priority_run and not args.encoder_run:
        parser.error("--encoder-equal-priority-run also requires --encoder-run")
    args.out.mkdir(parents=True,exist_ok=True)
    rows=list(read_rows(args.runs))
    write_csv(args.out/"operator_summary.csv",rows)
    figures(rows,args.out)
    if args.encoder_run:
        encoder = list(interference_rows(args.encoder_run))
        write_csv(args.out/"encoder_interference.csv",encoder)
        interference_figure(encoder,args.out)
    if args.encoder_equal_priority_run:
        comparison = list(compare_priorities(args.encoder_run,args.encoder_equal_priority_run))
        write_csv(args.out/"encoder_priority_sensitivity.csv",comparison)
        priority_figure(comparison,args.out)


if __name__=="__main__":
    main()
