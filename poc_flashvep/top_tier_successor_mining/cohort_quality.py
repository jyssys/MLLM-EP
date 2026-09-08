"""Faithful SERE batch-global primary-union quality control, not EP timing."""
import argparse
import json
import os
import random
import time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu

import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLMoeForConditionalGeneration

from evaluate_quality import set_policy
from quality_probe import Observer,append_jsonl,load_records,make_inputs


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--policies",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--gpu",type=int,default=3)
    ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--shards",type=int,default=1)
    ap.add_argument("--dataset",default="chartqa")
    ap.add_argument("--limit",type=int,default=128)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--max-pixels",type=int,default=1605632)
    ap.add_argument("--expert-parity",action="store_true")
    args=ap.parse_args()
    assert 0<=args.shard<args.shards
    allowed_devices()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.manual_seed(7317)
    start=time.time()
    started=datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True,exist_ok=True)
    model=Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map=f"cuda:{args.gpu}",
        attn_implementation="sdpa",experts_implementation="grouped_mm").eval()
    proc=AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side="left"
    rows=load_records(args.data,args.dataset,"heldout",args.limit,0,1)
    if args.expert_parity:
        checks=[]
        for row in rows[:4]:
            inp,_=make_inputs(proc,row,model.device,max_pixels=args.max_pixels)
            model.set_experts_implementation("grouped_mm")
            grouped=model(**inp,use_cache=False,logits_to_keep=1).logits.float()
            model.set_experts_implementation("eager")
            eager=model(**inp,use_cache=False,logits_to_keep=1).logits.float()
            checks.append({"request_id":row["id"],"max_abs":float((grouped-eager).abs().max()),
                           "cosine":float(torch.nn.functional.cosine_similarity(grouped.flatten(),eager.flatten(),dim=0)),
                           "greedy_equal":bool((grouped.argmax(-1)==eager.argmax(-1)).all()),
                           "kl":float(torch.nn.functional.kl_div(grouped.log_softmax(-1),eager.softmax(-1),reduction="sum"))})
        model.set_experts_implementation("grouped_mm")
        (args.out/"native_expert_parity.json").write_text(json.dumps(checks,indent=2))
        print(json.dumps({"expert_parity":checks}),flush=True)
    observer=Observer(model,proc,args.out)
    policies=json.loads(args.policies.read_text())
    cache={}
    for offset in range(0,len(rows),args.batch_size):
        if (offset//args.batch_size)%args.shards!=args.shard:
            continue
        batch=rows[offset:offset+args.batch_size]
        texts,images=[],[]
        for row in batch:
            ims=[Image.open(p).convert("RGB") for p in row["images"]]
            content=[{"type":"image","image":im} for im in ims]
            content.append({"type":"text","text":row["question"]+"\nAnswer the question using a single word or phrase."})
            texts.append(proc.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True))
            images.extend(ims)
        inputs=proc(text=texts,images=images,padding=True,padding_side="left",return_tensors="pt",max_pixels=args.max_pixels).to(model.device)
        order=list(policies)
        random.Random(7317+offset).shuffle(order)
        for policy in order:
            set_policy(observer,policy,model.device,cache)
            generated=model.generate(**inputs,max_new_tokens=32,do_sample=False)
            records=[]
            for i,row in enumerate(batch):
                tokens=generated[i,inputs.input_ids.shape[1]:].tolist()
                if proc.tokenizer.eos_token_id in tokens:
                    tokens=tokens[:tokens.index(proc.tokenizer.eos_token_id)+1]
                records.append({"request_id":row["id"],"image_id":row["image_id"],"dataset":row["dataset"],
                                "policy":policy["name"],"policy_definition":policy,
                                "prediction":proc.tokenizer.decode(tokens,skip_special_tokens=True),
                                "answers":row["answers"],"output_tokens":tokens,
                                "batch_size":len(batch),"cohort":[r["id"] for r in batch],
                                "prompt_tokens":int(inputs.attention_mask[i].sum()),"physical_gpu":physical_gpu(args.gpu)})
            append_jsonl(args.out/f"predictions_{args.shard}.jsonl",records)
            print(json.dumps({"offset":offset,"batch_size":len(batch),"policy":policy["name"]}),flush=True)
    completion="completed.json" if args.shards==1 else f"completed_{args.shard}.json"
    (args.out/completion).write_text(json.dumps({"started_utc":started,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.time()-start,
        "physical_gpu":physical_gpu(args.gpu),"arguments":{k:str(v) for k,v in vars(args).items()},
        "runtime":"HF batch-quality control; not EP serving timing"},indent=2))


if __name__=="__main__":main()
