"""Paired greedy task evaluation of unchanged paper-baseline algorithms.

Single-GPU HF mathematical port, NOT an EP speed benchmark. Policies are supplied
in a JSON list so exact calibration provenance is saved with every observation.
No policy is selected using held-out benchmark answers.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu

import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from quality_probe import Observer, append_jsonl, load_records, make_inputs


def set_policy(observer, policy, device, cache):
    observer.mode = policy["method"]
    observer.ablate_layer = -1
    observer.record = False
    observer.skip_counts = [0,0]
    observer.policy_phase = policy.get("phase","all")
    if policy["method"] == "sere":
        path = policy["similarity"]
        if path not in cache:
            cache[path] = torch.load(path,map_location=device,weights_only=True)
        observer.similarity = cache[path]
        observer.retain, observer.rho = policy.get("retain",2), policy.get("rho",0.5)
    elif policy["method"] == "modes":
        path = policy["alpha"]
        if path not in cache:
            cache[path] = torch.tensor(json.loads(Path(path).read_text())["alpha"], device=device)
        observer.alpha = cache[path]
        observer.tau_text, observer.tau_vision = policy["tau_text"], policy["tau_vision"]
    elif policy["method"] != "vanilla":
        raise ValueError(policy)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--policies",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--gpu",type=int,default=0,choices=range(4))
    ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--shards",type=int,default=1)
    ap.add_argument("--limit",type=int,default=256)
    ap.add_argument("--dataset",default="gqa")
    ap.add_argument("--split",default="heldout")
    ap.add_argument("--max-pixels",type=int,default=1605632)
    ap.add_argument("--max-new-tokens",type=int,default=32)
    ap.add_argument("--teacher-kl",action="store_true")
    args=ap.parse_args()
    allowed_devices()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.manual_seed(7317)
    start=time.time()
    started=datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True,exist_ok=True)
    policies=json.loads(args.policies.read_text())
    model=Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map=f"cuda:{args.gpu}",
        attn_implementation="sdpa",experts_implementation="grouped_mm").eval()
    processor=AutoProcessor.from_pretrained(args.model)
    observer=Observer(model,processor,args.out)
    cache={}
    rows=[row for dataset in args.dataset.split(",")
          for row in load_records(args.data,dataset,args.split,args.limit,args.shard,args.shards)]
    (args.out/f"environment_{args.shard}.json").write_text(json.dumps({
        "started_utc":started,"arguments":{k:str(v) for k,v in vars(args).items()},
        "policies":policies,"physical_gpu":physical_gpu(args.gpu),
        "timing_interpretation":"HF single-GPU correctness/quality only, not efficient implementation nor EP E2E"},indent=2))
    # Identical vanilla warmup before randomized per-request comparisons.
    warm,_=make_inputs(processor,rows[0],model.device,max_pixels=args.max_pixels)
    model.generate(**warm,max_new_tokens=2,do_sample=False)
    for index,row in enumerate(rows):
        inputs,prompt_len=make_inputs(processor,row,model.device,max_pixels=args.max_pixels)
        teacher=teacher_inputs=teacher_prompt=None
        if args.teacher_kl:
            set_policy(observer,{"method":"vanilla"},model.device,cache)
            teacher_inputs,teacher_prompt=make_inputs(processor,row,model.device,answer=True,max_pixels=args.max_pixels)
            teacher=model(**teacher_inputs,use_cache=False).logits[:,teacher_prompt-1:-1].float().softmax(-1)
        order=list(policies)
        random.Random(7317+index*args.shards+args.shard).shuffle(order)
        for policy in order:
            set_policy(observer,policy,model.device,cache)
            kl=None
            if teacher is not None:
                logits=model(**teacher_inputs,use_cache=False).logits[:,teacher_prompt-1:-1].float()
                kl=float(torch.nn.functional.kl_div(logits.log_softmax(-1),teacher,reduction="sum")/teacher.shape[1])
                observer.skip_counts=[0,0]
            torch.cuda.synchronize()
            t0=time.perf_counter()
            generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t0
            tokens=generated[0,prompt_len:].tolist()
            output=processor.tokenizer.decode(tokens,skip_special_tokens=True)
            record={"request_id":row["id"],"image_id":row["image_id"],"dataset":row["dataset"],
                    "policy":policy["name"],"policy_definition":policy,"prediction":output,
                    "answers":row["answers"],"prompt_tokens":prompt_len,"output_tokens":tokens,
                    "answer_prediction_kl":kl,"changed_or_skipped_assignments":observer.skip_counts[0],
                    "counted_assignments":observer.skip_counts[1],"diagnostic_wall_seconds":elapsed,
                    "physical_gpu":physical_gpu(args.gpu),"order": [p["name"] for p in order]}
            append_jsonl(args.out/f"predictions_{args.shard}.jsonl",[record])
            print(json.dumps({"index":index,"request":row["id"],"policy":policy["name"],
                              "prediction":output,"answer":row["answers"],"kl":kl}),flush=True)
    (args.out/f"completed_{args.shard}.json").write_text(json.dumps({
        "started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":time.time()-start,"requests":len(rows),"policies":len(policies)}))


if __name__ == "__main__":
    main()
