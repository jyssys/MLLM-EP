"""Execute the unchanged official MoDES frontier search with HF boundary port.

Each rank is a quality replica; optional Gloo reductions implement the reference
DDP calibration statistic. This does not benchmark EP serving performance.
"""
import argparse
import ast
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from quality_probe import Observer, append_jsonl, load_records, make_inputs


def official_search(source):
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="minimize_g_subject_to_f")
    env={"logger":SimpleNamespace(info=lambda x:None)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),env)
    return env[node.name]


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--alpha",type=Path,required=True)
    ap.add_argument("--official-source",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=1024)
    ap.add_argument("--grid",type=int,default=100)
    ap.add_argument("--targets",default="0.5,0.7,0.8,0.9")
    ap.add_argument("--gpu",type=int,default=3)
    ap.add_argument("--max-pixels",type=int,default=1605632)
    ap.add_argument("--dataset",default="gqa")
    ap.add_argument("--split",default="calibration")
    ap.add_argument("--logit-mask",choices=["official","answer_prediction"],default="official")
    args=ap.parse_args()
    devices = allowed_devices()
    world=int(os.environ.get("WORLD_SIZE","1"))
    rank=int(os.environ.get("RANK","0"))
    gpu=int(os.environ["LOCAL_RANK"]) if world>1 else args.gpu
    if world>1:
        dist.init_process_group("gloo")
    torch.cuda.set_device(gpu)
    torch.set_num_threads(4)
    torch.manual_seed(7317)
    start=time.time()
    started=datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True,exist_ok=True)
    model=Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map=f"cuda:{gpu}",
        attn_implementation="sdpa",experts_implementation="grouped_mm").eval()
    processor=AutoProcessor.from_pretrained(args.model)
    observer=Observer(model,processor,args.out)
    observer.alpha=torch.tensor(json.loads(args.alpha.read_text())["alpha"],device=model.device)
    observer.record=False
    observer.defer_skip_counts=True
    records=load_records(args.data,args.dataset,args.split,args.limit,rank,world)
    cached=[]
    for index,row in enumerate(records):
        inputs,prompt_len=make_inputs(processor,row,model.device,answer=True,
                                      max_pixels=args.max_pixels,instruction=False)
        begin=(int((inputs.input_ids[0]==151644).nonzero()[-1])+3
               if args.logit_mask=="official" else prompt_len-1)
        end=None if args.logit_mask=="official" else -1
        logits=model(**inputs,use_cache=False).logits[:,begin:end].float()
        cached.append((inputs.to("cpu"),logits.softmax(-1).cpu(),begin,end))
        if index%32==0:
            print(json.dumps({"stage":"teacher_cache","rank":rank,"completed":index+1}),flush=True)
    observer.mode="modes"
    cache={}
    existing=args.out/"evaluations.jsonl"
    if existing.exists():
        for line in existing.read_text().splitlines():
            r=json.loads(line)
            cache[(r["tau_text"],r["tau_vision"])]=(r["kl"],r["skip_fraction"])
    def evaluate(tt,tv):
        if (tt,tv) in cache:
            return cache[tt,tv]
        observer.tau_text,observer.tau_vision=tt,tv
        observer.skip_counts=[0,0]
        loss,count=torch.zeros((),dtype=torch.float64,device=model.device),0
        t0=time.perf_counter()
        for inputs,teacher,begin,end in cached:
            moved={k:v.to(model.device) for k,v in inputs.items()}
            logits=model(**moved,use_cache=False).logits[:,begin:end].float()
            loss+=F.kl_div(logits.log_softmax(-1),teacher.to(model.device),reduction="sum").double()
            count+=teacher.shape[1]
        stats=torch.tensor([float(loss),count,int(observer.skip_counts[0]),observer.skip_counts[1]],dtype=torch.float64)
        if world>1:
            dist.all_reduce(stats)
        value=(float(stats[0]/stats[1]),float(stats[2]/stats[3]))
        cache[tt,tv]=value
        if rank==0:
            record={"tau_text":tt,"tau_vision":tv,"kl":value[0],"skip_fraction":value[1],
                    "evaluations":len(cache),"elapsed_seconds":time.perf_counter()-t0,
                    "answer_tokens":int(stats[1]),"total_assignments":int(stats[3])}
            append_jsonl(existing,[record])
            print(json.dumps(record),flush=True)
        return value
    # Exact published repo default rectified-sigmoid mapping and endpoints.
    if cache:
        key=next(iter(cache))
        prior=cache.pop(key)
        repeated=evaluate(*key)
        assert math.isclose(prior[0],repeated[0],rel_tol=5e-4,abs_tol=1e-6),(prior,repeated)
        assert prior[1]==repeated[1],(prior,repeated)
        if rank==0:
            (args.out/"resume_parity.json").write_text(json.dumps({"thresholds":key,
                "prior_kl_skip":prior,"repeated_kl_skip":repeated,
                "change":"Defer statistics CPU synchronization only; model math unchanged"},indent=2))
    text=[2/(1+math.exp(-(0.71*i/(args.grid-1)-1)*10)) for i in range(args.grid)]
    vision=[2/(1+math.exp(-(0.6*i/(args.grid-1)-1)*10)) for i in range(args.grid)]
    search=official_search(args.official_source)
    results=[]
    for target in [float(x) for x in args.targets.split(",")]:
        tt,tv,loss=search(text,vision,evaluate,target,False)
        results.append({"target_skip":target,"tau_text":tt,"tau_vision":tv,"kl":loss,
                        "actual_skip":evaluate(tt,tv)[1] if tt is not None else None})
        if rank==0:
            (args.out/"frontier.json").write_text(json.dumps(results,indent=2))
    if rank==0:
        (args.out/"completed.json").write_text(json.dumps({"arguments":{k:str(v) for k,v in vars(args).items()},
            "started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds":time.time()-start,"replica_count":world,"evaluated_points":len(cache),
            "physical_gpus":devices if world>1 else [devices[gpu]],
            "calibration_request_count":args.limit,"runtime":"HF quality replicas; not EP timing"},indent=2))
    if world>1:
        dist.destroy_process_group()


if __name__=="__main__":
    main()
