"""Reduce saved three-GPU open-loop trials with trace-matched comparisons."""
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np


def percentile(items, q):
    return float(np.percentile(items,q)) if items else None


def block(run):
    suffix = re.search(r"_r([2-9]\d*|1\d+)$", run)
    return int(suffix[1]) if suffix else 1


def union(intervals):
    combined = []
    for lo,hi in sorted(intervals):
        if hi<lo:
            raise ValueError("Inverted stage clock")
        if combined and lo<=combined[-1][1]:
            combined[-1]=(combined[-1][0],max(hi,combined[-1][1]))
        else:
            combined.append((lo,hi))
    return combined


def common_duration(a,b):
    total=0.
    i=j=0
    while i<len(a) and j<len(b):
        begin=max(a[i][0],b[j][0]);end=min(a[i][1],b[j][1])
        total+=max(0.,end-begin)
        if a[i][1]<b[j][1]:
            i+=1
        else:
            j+=1
    return total


def stratified_requests(requests):
    groups = defaultdict(list)
    for row in requests:
        image_count = row["image_count"]
        group = ("text only" if image_count == 0 else "one image" if image_count == 1 else
                 "2–4 images" if image_count <= 4 else "5–19 images" if image_count < 20 else "20+ images")
        groups[group].append(row)
        if row["source"].endswith("MMLongBench-Doc"):
            groups["MMLongBench-Doc"].append(row)
    results = {}
    for name,rows in sorted(groups.items()):
        completed = [row for row in rows if row["outcome"] in ("eos","cap")]
        results[name] = {"planned":len(rows), "completed":len(completed),
                         "input_rows_p50":percentile([row["input_rows"] for row in rows],50),
                         "output_rows_p50":percentile([len(row["output_ids"]) for row in completed],50),
                         "ttft_p95_ms":percentile([row["ttft_ms"] for row in completed],95),
                         "tpot_p95_ms":percentile([row["tpot_ms"] for row in completed],95),
                         "itl_p99_ms":percentile([gap for row in completed for gap in row["itl_ms"]],99),
                         "joint_slo_requests":sum(row["joint_slo"] for row in rows)}
    return results


def receipt(path):
    d=json.loads(path.read_text())
    req=d["requests"];steps=d["decode_steps"]
    e=union((r["e_gpu_begin"],r["e_end"]) for r in req if "e_gpu_begin" in r)
    runs=union((x["begin"],x["end"]) for x in steps)
    helpers=union((x["begin"],x["end"]) for x in steps if x["assisted"])
    time_used=sum(b-a for a,b in helpers)
    d_busy=sum(b-a for a,b in runs)
    snapshots=[x for x in steps if x.get("decode_gpu_enclosure_ms")]
    stage_ms = lambda a,b: [(r[b]-r[a])*1e3 for r in req if a in r and b in r]
    e_wait=stage_ms("arrived","e_start")
    e_service=stage_ms("e_start","e_end")
    p_wait = [(r["p_start"]-r.get("e_end",r["arrived"]))*1e3 for r in req if "p_start" in r]
    attention_share=[x["attention_gpu_enclosure_ms"]/x["decode_gpu_enclosure_ms"] for x in snapshots]
    live_kv_rows=[sum(sum(part) for part in x["shards"]) for x in steps]
    ready_wait=stage_ms("p_end","d_admission")
    step_groups=defaultdict(list)
    for step in steps:
        group=(step["batch"], step["encoder_active_at_launch"], step["assisted"])
        step_groups[group].append(1e3*(step["end"]-step["begin"]))
    service_groups=[{"prepared_batch":batch,"encoder_active_at_launch":active,"assisted":assist,
                     "count":len(samples),"p50_ms":percentile(samples,50),"p95_ms":percentile(samples,95)}
                    for (batch, active, assist), samples in sorted(step_groups.items())]
    profile_overlap={}
    if time_used:
        profile_overlap["assisted_decode_host_step_overlap_encoder_gpu_enclosure_fraction"]=common_duration(e,helpers)/time_used
    if e and d_busy:
        profile_overlap["all_decode_host_step_overlap_encoder_gpu_enclosure_fraction"]=common_duration(e,runs)/d_busy
    offsets=[(r["id"],round(r["planned"]-d["start_monotonic"],6)) for r in req]
    digest=hashlib.sha256(json.dumps(offsets).encode()).hexdigest()
    result={k:v for k,v in d.items() if k not in ("requests","decode_steps")}
    result.update({"run":path.parent.name,"cohort":path.stem,"roster_offset_sha256":digest,
                   "independent_restart_block":block(path.parent.name),
                   "attention_engine":d.get("attention_engine","eager"),
                   "vision_backend":d.get("vision_backend","stock-sdpa"),
                   "encoder_pipeline":d.get("encoder_pipeline",False),
                   "graph_step_count":sum(x.get("attention_backend")=="graph" for x in steps),
                   "encoder_service_p50_ms":percentile(e_service,50),"encoder_service_p95_ms":percentile(e_service,95),
                   "encoder_gpu_p95_ms":percentile([r["e_gpu_enclosure_ms"] for r in req if "e_gpu_enclosure_ms" in r],95),
                   "encoder_queue_p95_ms":percentile(e_wait,95),"prefill_queue_p95_ms":percentile(p_wait,95),
                   "prefill_service_p95_ms":percentile(stage_ms("p_start","p_end"),95),
                   "decode_queue_p95_ms":percentile(ready_wait,95),
                   "initial_handoff_p95_ms":percentile([r["initial_host_ms"] for r in req if "initial_host_ms" in r],95),
                   "input_rows_p50":percentile([r["input_rows"] for r in req],50),
                   "input_rows_p95":percentile([r["input_rows"] for r in req],95),
                   "output_rows_p50":percentile([len(r["output_ids"]) for r in req if r["output_ids"]],50),
                   "output_rows_p95":percentile([len(r["output_ids"]) for r in req if r["output_ids"]],95),
                   "output_rows_sum":sum(len(r["output_ids"]) for r in req),
                   "image_count_strata_and_long_document":stratified_requests(req),
                   "joint_slo_requests":sum(r["joint_slo"] for r in req),
                   "joint_slo_tokens_per_second":sum(len(r["output_ids"]) for r in req if r["joint_slo"])/d["makespan_seconds"],
                   "page_wait_request_count":sum(bool(r.get("page_waits")) for r in req),
                   "page_wait_attempts":sum(r.get("page_waits",0) for r in req),
                   "decode_host_step_groups":service_groups,
                   "p_to_d_bytes":sum(r.get("p_to_d_logical_bytes",0) for r in req),
                   "p_to_helper_bytes":sum(r.get("p_to_helper_logical_bytes",0) for r in req),
                   "helper_retained_bytes":sum(r.get("helper_retained_bytes",0) for r in req),
                   "recurring_D_helper_bytes":sum(x.get("D_helper_logical_bytes",0) for x in steps),
                   "remote_requests":sum(bool(r.get("helper_prompt_rows")) for r in req),
                   "encode_decode_overlap":profile_overlap,
                   "encode_decode_overlap_scope":"Common monotonic host enclosures. No kernel-exclusive or future-safe claim.",
                   "profile_attention_share_p50":percentile(attention_share,50),
                   "profile_attention_share_p95":percentile(attention_share,95),
                   "profile_witness_steps":len(snapshots),
                   "max_prepared_batch":max((x["batch"] for x in steps),default=0),
                   "live_kv_rows_p50":percentile(live_kv_rows,50),
                   "live_kv_rows_p95":percentile(live_kv_rows,95),
                   "live_kv_rows_max":max(live_kv_rows,default=0),
                   "steps_at_observed_controlled_crossover_rows":sum(n >= 262144 for n in live_kv_rows),
                   "decode_steps_count":len(steps),
                   "warmup_excluded":True})
    return result,d,offsets


def compare(items):
    out=[]
    for arm,raw,offsets in items:
        if arm["helper"]=="none":
            continue
        for base,reference,native_offset in items:
            if base["helper"]!="none" or base["topology"]!=arm["topology"]:
                continue
            fields=("family","workload","offered_qps","cap","gpu_profiler","attention_engine",
                    "vision_backend","encoder_pipeline","independent_restart_block")
            if any(base[k]!=arm[k] for k in fields) or len(offsets)!=len(native_offset):
                continue
            if any(a[0]!=b[0] or abs(a[1]-b[1])>2e-6 for a,b in zip(offsets,native_offset,strict=True)):
                continue
            exact=sum(a["output_ids"]==b["output_ids"] for a,b in zip(raw["requests"],reference["requests"],strict=True)
                      if a["output_ids"] and b["output_ids"])
            latencies={k:(base["latencies"][k][q],arm["latencies"][k][q]) for k,q in
                       (("ttft_ms","p95"),("tpot_ms","p95"),("itl_ms","p99"),("e2e_ms","p95"))}
            out.append({"baseline":base["run"],"arm":arm["run"],"offered_qps":arm["offered_qps"],
                        "independent_restart_block":arm["independent_restart_block"],
                        "trace_rows":len(offsets),"baseline_or_arm_failed_requests":
                        [base["planned"]-base["complete"],arm["planned"]-arm["complete"]],
                        "natural_or_fixed_exact_output_ids":exact,
                        "latency_change_fraction":{k:(b/a-1 if a and b is not None else None)
                                                   for k,(a,b) in latencies.items()},
                        "joint_slo":[base["joint_slo"],arm["joint_slo"]],
                        "goodput":[base["goodput_requests_per_second"],arm["goodput_requests_per_second"]],
                        "encoder_gpu_p95_ms":[base["encoder_gpu_p95_ms"],arm["encoder_gpu_p95_ms"]]})
    return out


def summarize_restarts(pairs, items):
    cases = [row for row in pairs if row["baseline"].startswith("img_local_q25")
             and row["arm"].startswith("img_encoder_q25")]
    if not cases:
        return None
    changes = defaultdict(list)
    for row in cases:
        for metric, value in row["latency_change_fraction"].items():
            if value is not None:
                changes[metric].append(value)
        baseline, candidate = row["goodput"]
        if baseline:
            changes["goodput"].append(candidate / baseline - 1)
    locals_ = [(trial["run"], raw["requests"], offsets) for trial, raw, offsets in items
               if trial["run"].startswith("img_local_q25")]
    repeat_agreement = []
    for (left_name, left, left_trace), (right_name, right, right_trace) in combinations(locals_,2):
        if len(left_trace)!=len(right_trace) or any(
                a[0]!=b[0] or abs(a[1]-b[1])>2e-6 for a,b in zip(left_trace,right_trace,strict=True)):
            continue
        exact=sum(a["output_ids"]==b["output_ids"] and bool(a["output_ids"])
                  for a,b in zip(left,right,strict=True))
        repeat_agreement.append({"left":left_name,"right":right_name,"exact":exact,"planned":len(left)})
    return {"scope":"Matched independent server-process restarts. Range is across restart pairs.",
            "restart_blocks":[row["independent_restart_block"] for row in cases],
            "local_AA_exact_natural":repeat_agreement,
            "paired_local_assisted_exact_natural":[row["natural_or_fixed_exact_output_ids"] for row in cases],
            "relative_changes":{key:{"paired":values,"median":percentile(values,50),
                                      "minimum":min(values),"maximum":max(values),
                                      "sample_sd":float(np.std(values,ddof=1)) if len(values)>1 else None}
                                for key,values in changes.items()}}


def http_control(path):
    """Reduce the stock HTTP/SSE clients at the historical service-level boundaries."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    tag = path.stem.removesuffix("_requests")
    completion = json.loads((path.parent.parent / "state" / (tag + "_complete.json")).read_text())
    valid = [row for row in rows if row["status"]=="MEASURED_ONLINE"]
    qualified = [r for r in valid if r["TTFT_ms"]<=5000 and r["TPOT_ms"]<=33 and max(r["ITL_ms"],default=0)<=34]
    duration = completion["duration_seconds"]
    metrics = {name:[r[name] for r in valid] for name in ("TTFT_ms","TPOT_ms","E2E_ms")}
    metrics["ITL_ms"] = [gap for r in valid for gap in r["ITL_ms"]]
    return {"run":tag,"root":path.parent.parent.name,"raw_sha256":completion["raw_SHA256"],
            "boundary":"open-loop HTTP/SSE receiver",
            "planned":completion["planned_count"],"complete":len(valid),"joint_slo_requests":len(qualified),
            "requests_per_second":len(valid)/duration,"goodput_requests_per_second":len(qualified)/duration,
            "tokens_per_second":sum(r["output_tokens"] for r in valid)/duration,
            "goodput_tokens_per_second":sum(r["output_tokens"] for r in qualified)/duration,
            "latencies":{name:{f"p{q}":percentile(values,q) for q in (50,95,99)} for name,values in metrics.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",required=True,nargs="+",type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--http-raw",nargs="+",type=Path,help="Optional raw JSONL for stock HTTP controls")
    args=p.parse_args()
    paths=[name for run in args.runs for name in sorted(run.glob("cohort_*.json"))]
    items=[receipt(path) for path in paths]
    if not items:
        p.error("No measured cohorts")
    args.out.mkdir(parents=True,exist_ok=True)
    paired=compare(items)
    with (args.out/"summary.json").open("w") as stream:
        json.dump({"trials":[item[0] for item in items],"paired":paired,
                   "medium_load_independent_restart_summary":summarize_restarts(paired,items)},stream,indent=2)
        stream.write("\n")
    if args.http_raw:
        (args.out/"http_controls.json").write_text(json.dumps([http_control(path) for path in args.http_raw],indent=2)+"\n")
    with (args.out/"metrics.csv").open("w",newline="") as stream:
        keys=("run","family","workload","topology","helper","partition","attention_engine","graph_step_count","gpu_profiler","offered_qps","complete","planned","requests_per_second",
              "tokens_per_second","goodput_requests_per_second","joint_slo_tokens_per_second","joint_slo","max_prepared_batch","remote_requests",
              "encoder_gpu_p95_ms","encoder_queue_p95_ms","prefill_service_p95_ms","decode_queue_p95_ms",
              "initial_handoff_p95_ms","p_to_d_bytes","p_to_helper_bytes","recurring_D_helper_bytes",
              "input_rows_p50","input_rows_p95","output_rows_p50","output_rows_p95","page_wait_attempts",
              "profile_attention_share_p50","live_kv_rows_p95","live_kv_rows_max","steps_at_observed_controlled_crossover_rows")
        counters=("ttft_ms_p95","tpot_ms_p95","itl_ms_p99","e2e_ms_p95")
        writer=csv.DictWriter(stream,fieldnames=keys+counters,lineterminator="\n")
        writer.writeheader()
        for value,_,_ in items:
            row={key:value[key] for key in keys}
            for key in counters:
                metric,target=key.rsplit("_",1)
                row[key]=value["latencies"][metric][target]
            writer.writerow(row)


if __name__=="__main__":
    main()
