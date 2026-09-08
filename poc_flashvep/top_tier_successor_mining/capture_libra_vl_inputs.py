"""Exact natural equal-length VL groups and bounded native-bridge inputs.

CPU preparation does not reserve a GPU. Captures are real-image full Qwen-VL
forwards; recorded wall time includes instrumentation and is not a speed claim.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--manifest",type=Path)
    ap.add_argument("--only-group")
    ap.add_argument("--hidden-diagnostic",action="store_true")
    ap.add_argument("--groups",type=int,default=8)
    ap.add_argument("--max-pixels",type=int,default=1605632)
    ap.add_argument("--gpu",type=int,default=0,choices=range(4))
    ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--shards",type=int,default=1)
    args=ap.parse_args()
    allowed_devices()
    from transformers import AutoProcessor
    from quality_probe import make_inputs
    proc=AutoProcessor.from_pretrained(args.model)
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_path=args.manifest or args.out/"workload_manifest.json"
    if args.prepare_only:
        rows=[json.loads(line) for line in args.data.read_text().splitlines()]
        rng=random.Random(4917)
        groups=[]
        for dataset in ("gqa","chartqa"):
            pool=[r for r in rows if r["dataset"]==dataset and r["split"]=="heldout"]
            rng.shuffle(pool)
            buckets=defaultdict(list)
            selected=0
            for index,row in enumerate(pool):
                inputs,length=make_inputs(proc,row,"cpu",max_pixels=args.max_pixels)
                bucket=buckets[length]
                if row["image_id"] not in [r["image_id"] for r in bucket]:
                    bucket.append(row)
                if len(bucket)==4:
                    groups.append({"id":f"{dataset}_{selected}","dataset":dataset,
                        "M_per_source":length,"requests":list(bucket),
                        "selection":"NATURAL_EXACT_LENGTH_DISTINCT_IMAGE_NO_PADDING"})
                    buckets.pop(length)
                    selected+=1
                if index%32==0:
                    print(json.dumps({"dataset":dataset,"processed":index+1,"groups":selected}),flush=True)
                if selected>=args.groups//2:
                    break
            if selected<args.groups//2:
                print(json.dumps({"warning":"EXACT_LENGTH_GROUPS_INSUFFICIENT","dataset":dataset,"groups":selected}),flush=True)
        manifest_path.write_text(json.dumps({"model":args.model,"max_pixels":args.max_pixels,"groups":groups},indent=2))
        return
    import torch
    from transformers import Qwen3VLMoeForConditionalGeneration
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.manual_seed(7317)
    start=time.time();started=datetime.now(timezone.utc).isoformat()
    manifest=json.loads(manifest_path.read_text())
    assert manifest["model"]==args.model and manifest["max_pixels"]==args.max_pixels
    model=Qwen3VLMoeForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation="sdpa",experts_implementation="grouped_mm",device_map=f"cuda:{args.gpu}").eval()
    captured={}
    debug=None
    if args.hidden_diagnostic:
        from libra_vl_diagnostics import install_hidden_diagnostic
        debug=install_hidden_diagnostic(model.model.language_model.layers)
    def pre(module,positional,kw):
        captured["inputs_embeds"]=kw["inputs_embeds"].detach().clone().squeeze(0)
        captured["vision_mask"]=kw["visual_pos_masks"].detach().clone().squeeze(0)
        captured["deepstack"]=[t.detach().clone() for t in kw["deepstack_visual_embeds"]]
    def rope(module,inputs,output):
        captured["cos"],captured["sin"]=[t.detach().clone().squeeze(0) for t in output]
    model.model.language_model.register_forward_pre_hook(pre,with_kwargs=True)
    model.model.language_model.rotary_emb.register_forward_hook(rope)
    def route(layer):
        def hook(module,inputs):
            hidden,ids,weights=inputs
            captured.setdefault("routes",{})[layer]={"ids":ids.detach().clone(),"weights":weights.detach().clone()}
        return hook
    for index,layer in enumerate(model.model.language_model.layers):
        layer.mlp.experts.register_forward_pre_hook(route(index))
    items=[(g,i,r) for g in manifest["groups"] for i,r in enumerate(g["requests"])]
    if args.only_group:items=[item for item in items if item[0]["id"]==args.only_group]
    selected=items[args.shard::args.shards]
    with torch.inference_mode():
        for group,source,row in selected:
            captured.clear()
            if debug is not None:debug["active"]=True;debug["records"].clear()
            inputs,length=make_inputs(proc,row,model.device,max_pixels=args.max_pixels)
            assert length==group["M_per_source"]
            result=model(**inputs,use_cache=False,logits_to_keep=1)
            captured["reference_logits"]=result.logits[0,-1].float().detach().clone()
            captured["input_ids"]=inputs.input_ids[0].detach().clone()
            if debug is not None:
                debug["active"]=False
                captured["hidden_diagnostic"]=debug["records"]
            torch.cuda.synchronize()
            def cpu(obj):
                if isinstance(obj,torch.Tensor):return obj.cpu()
                if isinstance(obj,dict):return {k:cpu(v) for k,v in obj.items()}
                if isinstance(obj,list):return [cpu(v) for v in obj]
                return obj
            output={**cpu(captured),"request":row,"group":group["id"],"source_rank":source,
                    "M":length,"model":args.model,"capture_scope":"EXACT_VL_PREFILL_MATERIALIZED_INPUTS"}
            torch.save(output,args.out/f"{group['id']}_source{source}.pt")
            print(json.dumps({"request":row["id"],"group":group["id"],"source":source,"M":length,
                              "first_token":int(captured["reference_logits"].argmax())}),flush=True)
    (args.out/f"completed_{args.shard}.json").write_text(json.dumps({"started_utc":started,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.time()-start,
        "physical_gpu":physical_gpu(args.gpu),"requests":len(selected),"kind":"BRIDGE_PARITY_CAPTURE_NOT_PERFORMANCE"},indent=2))


if __name__=="__main__":main()
