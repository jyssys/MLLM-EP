"""Streaming accumulation of SERE's all-expert Frobenius calibration.

Avoids retaining [400*128, H] outputs for all experts simultaneously. Summed
squared pairwise distances are exactly additive across token chunks in real
arithmetic; normalize only after all calibration samples, matching one full batch.
FP32 accumulation is explicitly recorded and compared to official BF16 norm.
"""
import argparse
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3VLMoeForConditionalGeneration


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--text-model", action="store_true")
    ap.add_argument("--norm-semantics",choices=["fp32_gram","official_bf16"],default="fp32_gram")
    ap.add_argument("--vision-recalibration",action="store_true",
                    help="Trivial-fix control: same number of real Vision token inputs, original similarity math")
    args = ap.parse_args()
    allowed_devices()
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started=datetime.now(timezone.utc).isoformat()
    cls = AutoModelForCausalLM if args.text_model else Qwen3VLMoeForConditionalGeneration
    model = cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}",
                               attn_implementation="sdpa", experts_implementation="grouped_mm").eval()
    layers = model.model.layers if args.text_model else model.model.language_model.layers
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(s) for s in args.data.read_text().splitlines()]
    if args.vision_recalibration:
        assert not args.text_model
        from transformers import AutoProcessor
        from quality_probe import make_inputs
        import random
        processor=AutoProcessor.from_pretrained(args.model)
        chosen=[r for r in records if r["dataset"]=="gqa" and r["split"]=="calibration"]
        random.Random(7317).shuffle(chosen)
    else:
        chosen = [r for r in records if len(tokenizer.encode(r["text"])) > args.length][:args.count]
    assert len(chosen)>=args.count if args.vision_recalibration else len(chosen)==args.count
    distance_sums = torch.zeros(len(layers), 128,128,device=model.device,dtype=torch.float64)
    token_counts = [0]*len(layers)
    parity = []
    selected_positions = None
    position_records = []

    def hook(layer):
        def capture(module, inputs):
            h = inputs[0]
            if selected_positions is not None:
                h=h[selected_positions]
            gate_up = torch.bmm(h.unsqueeze(0).expand(128,-1,-1), module.gate_up_proj.transpose(1,2))
            gate, up = gate_up.chunk(2,dim=-1)
            activated = F.silu(gate)*up
            output = torch.bmm(activated,module.down_proj.transpose(1,2))
            if args.norm_semantics=="official_bf16":
                flat=output.flatten(1)
                sqdist=torch.empty(128,128,device=output.device,dtype=torch.float32)
                for offset in range(0,128,4):
                    # Match official rounded BF16 subtraction, then accumulate
                    # squared norms over token chunks. Final full norm is cast
                    # to BF16 just as torch.norm(BF16) before FP32 table storage.
                    diff=flat[offset:offset+4,None,:]-flat[None,:,:]
                    norms=torch.linalg.vector_norm(diff,dim=-1,dtype=torch.float32)
                    sqdist[offset:offset+4]=norms.square()
                    del diff,norms
            else:
                flat = output.flatten(1).float()
                norm = (flat*flat).sum(-1)
                sqdist = (norm[:,None]+norm[None,:]-2*(flat@flat.T)).clamp_min(0)
            sqdist.fill_diagonal_(0)
            distance_sums[layer].add_(sqdist.double())
            token_counts[layer] += h.shape[0]
            if token_counts[layer] == h.shape[0] and layer in [0,12,24,47]:
                # Compare actual official norm semantics on several expert pairs.
                pairs = []
                for a,b in [(0,1),(7,91),(63,127)]:
                    official = torch.norm(output[a]-output[b],p="fro").float()
                    streaming = sqdist[a,b].sqrt()
                    pairs.append({"a":a,"b":b,"official_norm":float(official),
                                  "streamed_norm":float(streaming),
                                  "relative_difference":float((official-streaming).abs()/official.clamp_min(1e-6))})
                parity.append({"layer":layer,"pairs":pairs})
        return capture
    handles = [layer.mlp.experts.register_forward_pre_hook(hook(l)) for l,layer in enumerate(layers)]
    if args.vision_recalibration:
        rng=torch.Generator(device=model.device).manual_seed(7317)
        used=[]
        for row in chosen:
            inp,_=make_inputs(processor,row,model.device,max_pixels=1605632,instruction=False)
            positions=(inp.input_ids[0]==model.config.image_token_id).nonzero().flatten()
            if positions.numel()<args.length:
                continue
            selected_positions=positions[torch.randperm(positions.numel(),generator=rng,device=model.device)[:args.length]]
            position_records.append({"id":row["id"],"image_id":row["image_id"],"positions":selected_positions.tolist()})
            model(**inp,use_cache=False,logits_to_keep=1)
            used.append(row)
            index=len(used)-1
            if index%8==0:
                print(json.dumps({"completed_sequences":index+1,"elapsed_seconds":time.time()-t0}),flush=True)
            if len(used)==args.count:
                break
        chosen=used
        assert len(chosen)==args.count
    else:
        for offset in range(0,len(chosen),args.batch_size):
            chunk = chosen[offset:offset+args.batch_size]
            inp = tokenizer([r["text"] for r in chunk],return_tensors="pt",padding=True,
                            truncation=True,max_length=args.length).to(model.device)
            model(**inp,use_cache=False,logits_to_keep=1)
            print(json.dumps({"completed_sequences":offset+len(chunk),"elapsed_seconds":time.time()-t0}),flush=True)
    distance = distance_sums.sqrt()
    if args.norm_semantics=="official_bf16":
        distance=distance.to(torch.bfloat16).float()
    similarity = (1-distance/distance.amax(dim=(-1,-2),keepdim=True).clamp_min(1e-12)).float()
    torch.save(similarity.cpu(),args.out/"similarity.pt")
    torch.save(distance_sums.cpu(),args.out/"pairwise_squared_distance_sums.pt")
    summary = {"model":args.model,"count":len(chosen),"length":args.length,
               "physical_gpu":physical_gpu(args.gpu),"sample_ids":[r["id"] for r in chosen],
               "started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
               "norm_semantics":args.norm_semantics,
               "token_counts":token_counts,"elapsed_seconds":time.time()-t0,
               "method":"SERE Frobenius, full-batch normalization after streamed squared distances",
               "vision_recalibration":args.vision_recalibration,
               "selected_vision_positions":position_records,
               "official_norm_parity":parity}
    (args.out/"calibration.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"status":"COMPLETE","elapsed_seconds":time.time()-t0}),flush=True)


if __name__ == "__main__":
    main()
