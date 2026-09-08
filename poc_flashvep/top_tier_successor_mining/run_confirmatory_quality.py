"""Fixed held-out quality screen after official calibration, no speed claims."""
import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    args=parser.parse_args()
    devices=allowed_devices()
    root=args.results.resolve();task=Path(__file__).resolve().parent
    assert (root/"quality/modes_frontier1024_grid100/completed.json").exists()
    policies=json.loads((task/"SERE/confirmatory_policies.json").read_text())
    policies += [p for p in json.loads((root/"analysis/modes_confirmatory_policies.json").read_text())
                 if p["method"]!="vanilla"]
    for p in policies:
        if "similarity" in p:p["similarity"]=str(Path(p["similarity"]).resolve())
    policy_file=root/"analysis/combined_confirmatory_policies_20260908.json"
    policy_file.write_text(json.dumps(policies,indent=2))
    py="/home/esjung/.venvs/top-tier-successor-quality/bin/python"
    model="/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
         "HF_HUB_DISABLE_PROGRESS_BARS":"1","OMP_NUM_THREADS":"4"}
    data=str(root/"data/requests.jsonl")
    common=["--model",model,"--data",data,"--limit","128"]
    b1=root/"quality/official_confirmatory_b1_20260908"
    chart=root/"quality/sere_official_chart128_b16_20260908"
    gqa=root/"quality/sere_official_gqa128_b16_20260908"
    for out in (b1,chart,gqa):assert not out.exists(),out

    def group(label,commands,timeout=3600):
        children=[];handles=[];start=time.monotonic()
        print(json.dumps({"stage":label,"started_utc":datetime.now(timezone.utc).isoformat(),
                          "physical_gpus":devices,"commands":commands}),flush=True)
        try:
            for i,command in enumerate(commands):
                handle=(root/f"raw/{label}_{i}.log").open("w");handles.append(handle)
                children.append(subprocess.Popen(command,env=env,stdout=handle,
                    stderr=subprocess.STDOUT,start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None,0) for p in children):
                    raise RuntimeError(f"{label}: child failed; inspect log, no automatic retry")
                if time.monotonic()-start>timeout:raise TimeoutError(label)
                time.sleep(2)
            assert all(p.returncode==0 for p in children)
            print(json.dumps({"stage":label,"complete":True,"seconds":time.monotonic()-start}),flush=True)
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for p in children:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            for f in handles:f.close()

    group("confirmatory_b1_20260908",[[py,str(task/"evaluate_quality.py"),*common,
        "--dataset","chartqa,gqa","--policies",str(policy_file),"--out",str(b1),
        "--gpu",str(rank),"--shard",str(rank),"--shards","4"] for rank in range(4)])
    commands=[]
    for rank in range(4):
        dataset="chartqa" if rank<2 else "gqa"
        commands.append([py,str(task/"cohort_quality.py"),*common,
            "--dataset",dataset,"--policies",str(task/"SERE/confirmatory_policies.json"),
            "--out",str(chart if rank<2 else gqa),"--gpu",str(rank),
            "--shard",str(rank%2),"--shards","2","--batch-size","16"])
    group("sere_confirmatory_b16_20260908",commands)
    print("COMPLETE: agent must score and inspect before further policy selection",flush=True)


if __name__=="__main__":main()
