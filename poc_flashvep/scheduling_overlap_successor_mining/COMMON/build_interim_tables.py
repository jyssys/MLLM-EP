"""CPU-only derived request index and conservative experiment-time accounting."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def percentile(values, q):
    if not values:
        return ""
    values = sorted(values)
    k = (len(values)-1)*q
    lo = int(k)
    return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(k-lo)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    r = args.results
    request_rows, intervals = [], []

    def record_interval(name, candidate, start, end, kind, valid, artifact, question, gpus="4,5,6,7"):
        intervals.append(dict(run_id=name, candidate=candidate, research_question=question,
                              start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                              end_utc=datetime.fromtimestamp(end, timezone.utc).isoformat(),
                              physical_gpus=gpus, elapsed_seconds=end-start,
                              gpu_hours=len(gpus.split(","))*(end-start)/3600, kind=kind, valid=valid, artifact=str(artifact)))

    def request(name, candidate, policy, workload, restart, rid, ttft, tpot, itl, e2e, count, correctness, artifact):
        request_rows.append(dict(run_id=name, candidate=candidate, policy=policy, workload=workload,
                                 restart=restart, request_id=rid, ttft_ms=1000*ttft,
                                 tpot_ms=1000*tpot if tpot is not None else "",
                                 itl_p99_ms=1000*percentile(itl,.99) if itl else "",
                                 e2e_ms=1000*e2e, output_tokens=count, correctness=correctness,
                                 artifact=str(artifact)))

    for path in sorted((r/"fastpp_runs").rglob("run.json")):
        run = json.loads(path.read_text())
        if not run["status"].startswith("REQUEST_COLLECTION_COMPLETE"):
            continue
        for measured in run["request_runs"]:
            if not measured["label"].startswith("measure"):
                continue
            source = path.parent/(measured["label"]+".jsonl")
            name = f"{path.parent.parent.name}:{run['run_id']}:{measured['label']}"
            for row in map(json.loads, source.read_text().splitlines()):
                request(name, "FASTPP", run["variant"] + (":"+run["explicit_pp_partition"] if run.get("explicit_pp_partition") else ""), Path(measured["trace"]).stem, path.parent.name,
                        row["request_id"], row["ttft_s"], row["tpot_s"], row["itl_s"], row["e2e_s"],
                        row["output_tokens"], "INSTRUMENTED_NOT_CLEAN_PERFORMANCE" if run.get("mechanism_instrumented") else "SHORT_PASS_CONTINUATION_PENDING", source)
            if run.get("mechanism_instrumented"):
                record_interval(name,"FASTPP",measured["start_unix"],measured["start_unix"]+measured["elapsed_s"],
                                "mechanism_diagnostic","INSTRUMENTED_NOT_CLEAN_PERFORMANCE",source,
                                "Native Qwen3 PP4 ALP prediction and stage-cost transfer")
            else:
                record_interval(name,"FASTPP",measured["start_unix"],measured["start_unix"]+measured["elapsed_s"],
                                "measurement","CROSS_CONFIG_CORRECTNESS_PENDING",source,
                                "Native original scheduling and existing static-partition request control")

    for path in sorted((r/"layered_runs").rglob("run.json")):
        run = json.loads(path.read_text())
        if not run["status"].startswith("COLLECTED"):
            continue
        for measured in run["request_runs"]:
            if not measured["label"].startswith("measure"):
                continue
            source = path.parent/(measured["label"]+".jsonl")
            name = f"{path.parent.parent.name}:{path.parent.name}:{measured['label']}"
            policy = f"{run['mode']}:tokens{run['tokens']}:stages{run['stages']}:graph{run['cuda_graph']}"
            for row in map(json.loads, source.read_text().splitlines()):
                assert row["status"] == "PASS"
                request(name, "LAYERED_PREFILL", policy, Path(measured["trace"]).stem,
                        path.parent.name, row["request_id"], row["ttft_s"], row["tpot_s"],
                        row["itl_s"], row["e2e_s"], row["output_tokens"],
                        "SHORT_PASS_CONTINUATION_PENDING", source)

    # Both clocks are same-host CPU clocks, not CUDA timestamps. Only the UTC
    # placement is approximated; duration uses differences of original monotonic
    # request times. No overlapping DP/rank interval is summed twice.
    realtime_offset = time.time()-time.monotonic()
    for run_dir in sorted(path.parent for path in (r/"vl_transfer").rglob("manifest.json")):
        run_name = str(run_dir.relative_to(r/"vl_transfer"))
        groups = defaultdict(list)
        for path in run_dir.glob("requests_dp*.jsonl"):
            for row in map(json.loads,path.read_text().splitlines()):
                if row["warmup"]:
                    continue
                groups[row["label"]].append(row)
                times = row["token_times"]
                request(run_name,"COMMON_VL_TRANSFER","observer" if row["instrumented"] else "clean",
                        row["family"],run_dir.name,row["request_id"],row["ttft_s"],row["tpot_s"],
                        [b-a for a,b in zip(times,times[1:])],row["e2e_s"],len(row["output_ids"]),
                        "DIAGNOSTIC_NOT_NATIVE_SCHEDULER_PERFORMANCE",path)
        for label, group in groups.items():
            start, end = min(x["start_s"] for x in group), max(x["complete_s"] for x in group)
            record_interval(f"{run_name}:{label}","COMMON_VL_TRANSFER",start+realtime_offset,end+realtime_offset,
                            "mechanism_diagnostic" if group[0]["instrumented"] else "measurement",
                            "COHORT_TRANSFER_DIAGNOSTIC_UTC_PLACEMENT_APPROX",run_dir,
                            "Real-image operation cost and observer overhead; same-host request interval union")

    for manifest_path in sorted((r/"nanoflow_runs").glob("*/manifest.json")):
        cohort = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        if cohort.name != "cohort_pilot_v1" and not manifest.get("contract", "").startswith("real-content fixed cohorts"):
            continue
        pair_status = {}
        pair_file = cohort / "paired_results.csv"
        if pair_file.is_file():
            with pair_file.open() as stream:
                pair_status = {(int(row["block"]), row["workload"], row["variant"]): row["status"]
                               for row in csv.DictReader(stream)}
        for rec in manifest["records"]:
            source = cohort/(rec["name"]+".json")
            if rec.get("returncode") != 0 or not source.exists():
                continue
            for row in json.loads(source.read_text())["requests"]:
                request(cohort.name+":"+rec["name"],"NANOFLOW",rec["variant"],rec["workload"],rec["block"],row["request_id"],
                        row["ttft_s"],row["tpot_s"],row["itl_s"],row["e2e_s"],len(row["output_ids"]),
                        "PROFILED_NOT_PERFORMANCE" if manifest.get("nsight_profiled") else
                        pair_status.get((rec["block"], rec["workload"], rec["variant"]),
                                        "BASELINE_OR_UNPAIRED_SANITY_ONLY"), source)
    for hf_path in sorted((r/"nanoflow_runs").glob("hf*reference*/manifest.json")):
        for row in json.loads(hf_path.read_text())["runs"]:
            record_interval("nanoflow_hf_same_prefix:"+Path(row["screen"]).name, "NANOFLOW",
                            row["start_unix"], row["end_unix"], "correctness_diagnostic",
                            "INDEPENDENT_HF_REFERENCE_NOT_PERFORMANCE", hf_path,
                            "Same-prefix HF versus native MoE logits", gpus="4")
    numerical_roots = [path for path in (r/"nanoflow_runs").glob("numerical_*") if path.is_dir()]
    for run_dir in [r/"nanoflow_runs/mechanism_smoke_v1", r/"nanoflow_runs/mechanism_smoke_v2", *numerical_roots]:
        for path in run_dir.glob("*.run_meta.json"):
            meta = json.loads(path.read_text())
            if not meta["result_exists"]:
                continue
            # Initial load/JIT and final Terminate are conservatively excluded.
            for row in meta["step_timings"][1:-1]:
                record_interval(f"{run_dir.name}:{path.stem}:{row['step']}","NANOFLOW",
                                row["start_unix"],row["end_unix"],"correctness_diagnostic",
                                "NUMERICAL_DIAGNOSTIC_NOT_PERFORMANCE",path,"Native graph/split output correctness")
    meta_path = r/"raw/qwen25_hf_reference.meta.json"
    meta = json.loads(meta_path.read_text())
    if "inference_start_unix" in meta and "end_unix" in meta:
        record_interval("qwen25_hf_reference","FASTPP",meta["inference_start_unix"],meta["end_unix"],
                        "correctness_diagnostic","REFERENCE_ONLY",meta_path,"Independent dense HF correctness reference")

    output = root/"REQUEST_E2E_RESULTS.csv"
    with output.open("w") as file:
        writer=csv.DictWriter(file,fieldnames=list(request_rows[0]));writer.writeheader();writer.writerows(request_rows)
    timing = root/"GPU_TIME_LOG.csv"
    with timing.open() as file:
        reader=csv.DictReader(file);fields=reader.fieldnames;existing=list(reader)
    ids={x["run_id"] for x in existing}
    new=[x for x in intervals if x["run_id"] not in ids]
    assert len({x["run_id"] for x in new})==len(new)
    with timing.open("a") as file:
        csv.DictWriter(file,fieldnames=fields).writerows(new)
    all_rows=existing+new
    ranges=sorted((datetime.fromisoformat(x["start_utc"]).timestamp(),datetime.fromisoformat(x["end_utc"]).timestamp()) for x in all_rows)
    merged=[]
    for start,end in ranges:
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],end)
        else: merged.append([start,end])
    total=sum(end-start for start,end in merged)
    # Native Layered uses two GPUs, whereas the other runs normally use four.
    # Union each physical GPU independently, never multiply every interval by 4.
    per_gpu = defaultdict(list)
    for row in all_rows:
        start = datetime.fromisoformat(row["start_utc"]).timestamp()
        end = datetime.fromisoformat(row["end_utc"]).timestamp()
        for gpu in row["physical_gpus"].split(","):
            assert gpu in {"4", "5", "6", "7"}, row
            per_gpu[gpu].append((start, end))
    gpu_seconds = {}
    for gpu, spans in per_gpu.items():
        union = []
        for start, end in sorted(spans):
            if union and start <= union[-1][1]:
                union[-1][1] = max(end, union[-1][1])
            else:
                union.append([start, end])
        gpu_seconds[gpu] = sum(end-start for start, end in union)
    total_gpu_hours = sum(gpu_seconds.values()) / 3600
    summary={"request_rows":len(request_rows),"new_time_intervals":len(new),
             "recorded_live_experiment_seconds_union":total,
             "recorded_gpu_hours":total_gpu_hours,
             "four_gpu_equivalent_wall_hours":total_gpu_hours/4,
             "recorded_seconds_per_physical_gpu":gpu_seconds,
             "exclusions":"burn; model load; JIT/compilation; failed startups; unmeasured first Nano step; some early pilot diagnostics",
             "not_cuda_busy_time":True,"status":"CONSERVATIVE_RECORDED_COVERAGE_NOT_COMPLETE_DEVICE_UTILIZATION"}
    (r/"cpu_analysis/interim_accounting.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
