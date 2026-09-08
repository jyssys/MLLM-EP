"""Account completed GPU-resident experiment intervals, never idle burn.

Wall intervals include model loading / associated CPU control and are explicitly
not a sum of CUDA kernel durations. Telemetry separately records GPU activity.
"""
import argparse
import csv
import json
from datetime import datetime,timezone
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--since-utc",help="Optional resumed-turn boundary; clip intervals at this UTC instant")
    args=ap.parse_args()
    rows=[]
    files=list(args.results.glob("quality/**/completed*.json"))
    files+=list(args.results.glob("quality/**/interrupted_user_release.json"))
    files+=list(args.results.glob("quality/sere*/calibration.json"))
    files+=list(args.results.glob("online/**/completed*.json"))
    for f in sorted(set(files)):
        d=json.loads(f.read_text())
        if "elapsed_seconds" not in d and not (d.get("started_utc") and d.get("finished_utc")):
            continue
        shard=f.stem.removeprefix("completed_")
        envpath=f.with_name(f"environment_{shard}.json")
        env=json.loads(envpath.read_text()) if envpath.exists() else {}
        started=d.get("started_utc",env.get("started_utc"))
        ended=d.get("finished_utc")
        clock_source="recorded_UTC"
        if not started or not ended:
            end=f.stat().st_mtime
            ended=datetime.fromtimestamp(end,timezone.utc).isoformat()
            started=datetime.fromtimestamp(end-d["elapsed_seconds"],timezone.utc).isoformat()
            clock_source="completed_file_mtime_minus_recorded_elapsed"
        elapsed=d.get("elapsed_seconds",(datetime.fromisoformat(ended)-datetime.fromisoformat(started)).total_seconds())
        if args.since_utc:
            since=datetime.fromisoformat(args.since_utc)
            if datetime.fromisoformat(ended)<=since:continue
            if datetime.fromisoformat(started)<since:
                started=since.isoformat()
                elapsed=(datetime.fromisoformat(ended)-since).total_seconds()
        gpu=d.get("physical_gpu",env.get("physical_gpu"))
        if gpu is None:
            gpu=int(d.get("arguments",{}).get("gpu",3))+1
        explicit=d.get("physical_gpus")
        count=d.get("replica_count",len(explicit) if explicit else 1)
        # Old unannotated four-rank records used1–4. New records carry explicit
        # physical devices; never relabel historical runs with current policy.
        devices=";".join(map(str,explicit)) if explicit else "1;2;3;4" if count==4 else str(gpu)
        assert set(map(int,devices.split(";")))<=set(range(1,8)),devices
        kind=("metadata_operator_port_diagnostic" if "metadata_operator" in str(f) else
              "native_supplement_functional" if "libra_native" in str(f) and d.get("layers",4)<48 else
              "native_supplement_prefill" if "libra_native" in str(f) else
              "EP_serving" if "online" in f.parts else "quality_replica_not_EP_throughput")
        status=d.get("status", "PORT_FAILURE" if any(d.get("exit_codes",[0])) else "COMPLETE")
        if (f.parent/'INVALIDATED.json').exists():
            status='INVALIDATED_OR_PORT_FAILURE_REAL_GPU_TIME_NOT_SCIENTIFIC_RESULT'
        rows.append({"experiment":str(f.parent.relative_to(args.results)),"start_utc":started,"end_utc":ended,
                     "physical_devices":devices,"resident_experiment_seconds":elapsed,
                     "unadjusted_process_gpu_hours":elapsed*count/3600,"clock_source":clock_source,
                     "kind":kind,"status":status})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fields=["experiment","start_utc","end_utc","physical_devices","resident_experiment_seconds",
            "unadjusted_process_gpu_hours","clock_source","kind","status"]
    with args.output.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    totals={}
    for gpu in sorted({int(g) for r in rows for g in r["physical_devices"].split(";")}):
        intervals=sorted((datetime.fromisoformat(r["start_utc"]).timestamp(),datetime.fromisoformat(r["end_utc"]).timestamp())
                         for r in rows if str(gpu) in r["physical_devices"].split(";"))
        merged=[]
        for start,end in intervals:
            if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
            else:merged.append([start,end])
        totals[gpu]=sum(b-a for a,b in merged)/3600
    summary={"recorded_intervals":len(rows),
             "since_utc":args.since_utc,
             "completed_intervals":sum(r["status"]=="COMPLETE" for r in rows),
             "interrupted_intervals":sum(r["status"]=="INTERRUPTED_USER_GPU_RELEASE" for r in rows),
             "overlap_deduplicated_gpu_resident_hours":sum(totals.values()),
             "per_physical_gpu_resident_hours":totals,"burn_hours_counted":0,
             "note":"Recorded process intervals include loading and CPU control, not CUDA active duration. Explicit user-release interruption is included; other aborted jobs without timestamps and primitive tests are not counted here."}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__=="__main__":main()
