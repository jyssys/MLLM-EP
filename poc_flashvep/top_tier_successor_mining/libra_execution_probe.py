"""Real-input EP4 Libra mechanism diagnostic, NOT a serving speed reproduction.

Preserves exact assignments and weights. Uses literal paper replica planning,
remote-only sharding, AllGather dispatch, local-before-remote grouped Triton
compute, and real-weight SymmetricMemory peer copies. CPU plans are prepared
outside the CUDA timing: unavailable official Cython overhead is not fabricated.
"""
import argparse
import gc
import json
import os
import random
import time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from libra_paper_plan import plan,evaluate_plan


def load_weight(model,index,key,start,end):
    with safe_open(str(model/index[key]),framework="pt",device="cpu") as f:
        return f.get_slice(key)[start:end].contiguous()


def assignment_owners(ids,placement):
    """Materialize Algorithm 2's exact assignment quantities, stable token ties."""
    counts=np.stack([np.bincount(a[a>=0],minlength=128) for a in ids])
    state=evaluate_plan(counts,placement)
    owners=np.full(ids.shape,-1,dtype=np.int32)
    for source in range(4):
        valid=ids[source]>=0
        local=valid & placement[ids[source].clip(min=0),source]
        owners[source][local]=source
    for expert in range(128):
        locations=np.argwhere((ids==expert) & (owners<0))
        cursor=0
        for dest in range(4):
            n=int(state["remote"][dest,expert])
            chosen=locations[cursor:cursor+n]
            if n:
                owners[tuple(chosen.T)]=dest
            cursor+=n
        assert cursor==len(locations)
    assert ((owners>=0)==(ids>=0)).all()
    for source in range(4):
        valid=ids[source]>=0
        assert placement[ids[source][valid],owners[source][valid]].all()
    return owners,state


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--capture",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--layers",default="1,3,12,24,36,47")
    ap.add_argument("--groups",type=int,default=4)
    ap.add_argument("--reps",type=int,default=20)
    ap.add_argument("--warmup",type=int,default=5)
    ap.add_argument("--max-tokens",type=int,default=1024)
    args=ap.parse_args()
    allowed_devices()
    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl",device_id=torch.device(f"cuda:{rank}"))
    assert dist.get_world_size()==4
    started=datetime.now(timezone.utc).isoformat()
    symm.set_backend("CUDA")
    args.out.mkdir(parents=True,exist_ok=True)
    index=json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    routes=args.capture/"routes"
    request_ids=sorted(p.name.removesuffix("_prefill_l00.pt") for p in routes.glob("*_prefill_l00.pt"))
    assert len(request_ids)>=4,request_ids
    homes=np.eye(4,dtype=bool)[np.arange(128)//32]
    stream=torch.cuda.current_stream()
    comm=torch.cuda.Stream()
    copy=torch.cuda.Stream()
    results=[]
    for layer in [int(x) for x in args.layers.split(",")]:
        prefix=f"model.language_model.layers.{layer}.mlp.experts."
        w1cpu=load_weight(args.model,index,prefix+"gate_up_proj",rank*32,(rank+1)*32)
        w2cpu=load_weight(args.model,index,prefix+"down_proj",rank*32,(rank+1)*32)
        base1=symm.empty(w1cpu.shape,dtype=torch.bfloat16,device="cuda")
        base2=symm.empty(w2cpu.shape,dtype=torch.bfloat16,device="cuda")
        base1.copy_(w1cpu);base2.copy_(w2cpu)
        hdl1=symm.rendezvous(base1,dist.group.WORLD)
        hdl2=symm.rendezvous(base2,dist.group.WORLD)
        peers1=[hdl1.get_buffer(r,base1.shape,base1.dtype) for r in range(4)]
        peers2=[hdl2.get_buffer(r,base2.shape,base2.dtype) for r in range(4)]
        torch.cuda.synchronize();dist.barrier()
        for group_idx in range(min(args.groups,len(request_ids)//4)):
            group=request_ids[group_idx*4:group_idx*4+4]
            snapshots=[torch.load(routes/f"{rid}_prefill_l{layer:02d}.pt",map_location="cpu",weights_only=True) for rid in group]
            predictions=[torch.load(routes/f"{rid}_prefill_l{layer-1:02d}.pt",map_location="cpu",weights_only=True)["next_prediction"] for rid in group]
            assert all("hidden" in s for s in snapshots),"Fresh exact-input capture required"
            for modality in ["all","vision","text"]:
                selections=[np.flatnonzero(np.ones(len(s["ids"]),dtype=bool) if modality=="all" else s[modality].numpy())[:args.max_tokens] for s in snapshots]
                if not all(len(x)>0 for x in selections):
                    continue
                cap=max(map(len,selections))
                ids=np.full((4,cap,8),-1,dtype=np.int64)
                pred=np.full_like(ids,-1)
                for r,sel in enumerate(selections):
                    ids[r,:len(sel)]=snapshots[r]["ids"][sel].numpy()
                    pred[r,:len(sel)]=predictions[r][sel].numpy()
                demand=np.stack([np.bincount(a[a>=0],minlength=128) for a in ids])
                predicted=np.stack([np.bincount(a[a>=0],minlength=128) for a in pred])
                cpu_start=time.perf_counter()
                pp=plan(predicted);op=plan(demand)
                cpu_plan_ms=(time.perf_counter()-cpu_start)*1000
                placements={"home_single":homes,"home_split":homes,
                            "libra_predicted":pp["placement"],"libra_perfect_prediction":op["placement"]}
                hidden=torch.zeros(cap,2048,device="cuda",dtype=torch.bfloat16)
                source_weights=torch.zeros(cap,8,device="cuda",dtype=torch.float32)
                sel=selections[rank]
                hidden[:len(sel)]=snapshots[rank]["hidden"][sel].cuda()
                source_weights[:len(sel)]=snapshots[rank]["weights"][sel].float().cuda()
                all_hidden=torch.empty(cap*4,2048,device="cuda",dtype=torch.bfloat16)
                all_weights=torch.empty(cap*4,8,device="cuda",dtype=torch.float32)
                # Weights/route metadata are fixed snapshot data. Exclude their
                # transfer for all policies; this is an optimistic stage oracle.
                dist.all_gather_into_tensor(all_weights,source_weights)
                all_ids=torch.from_numpy(ids.reshape(-1,8)).cuda()
                source_ids=all_ids[rank*cap:(rank+1)*cap].contiguous()
                configs={}
                for name,placement in placements.items():
                    owners,state=assignment_owners(ids,placement)
                    experts=np.flatnonzero(placement[:,rank])
                    native=np.arange(rank*32,(rank+1)*32)
                    extra=experts[experts//32!=rank]
                    experts=np.concatenate([native,extra])
                    emap=torch.full((128,),-1,device="cuda",dtype=torch.int32)
                    emap[torch.tensor(experts,device="cuda")]=torch.arange(len(experts),device="cuda",dtype=torch.int32)
                    a=torch.empty((len(experts),*base1.shape[1:]),device="cuda",dtype=torch.bfloat16)
                    b=torch.empty((len(experts),*base2.shape[1:]),device="cuda",dtype=torch.bfloat16)
                    a[:32].copy_(base1);b[:32].copy_(base2)
                    c0,c1=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    with torch.cuda.stream(copy):
                        copy.wait_stream(stream);c0.record()
                        for slot,e in enumerate(extra,32):
                            a[slot].copy_(peers1[e//32][e%32],non_blocking=True)
                            b[slot].copy_(peers2[e//32][e%32],non_blocking=True)
                        c1.record()
                    stream.wait_event(c1)
                    c1.synchronize()
                    local=placement[ids[rank].clip(min=0),rank] & (ids[rank]>=0)
                    # Remote assignments exclude each token's own source-local
                    # work even when another replica exists on this rank.
                    remote=(owners==rank)
                    remote[rank]=False
                    configs[name]={"w1":a,"w2":b,"emap":emap,
                        "local_weights":source_weights.masked_fill(~torch.from_numpy(local).cuda(),0),
                        "remote_weights":all_weights.masked_fill(~torch.from_numpy(remote.reshape(-1,8)).cuda(),0),
                        "local_ids":source_ids.masked_fill(~torch.from_numpy(local).cuda(),-1),
                        "remote_ids":all_ids.masked_fill(~torch.from_numpy(remote.reshape(-1,8)).cuda(),-1),
                        "copy_ms":c0.elapsed_time(c1),"extra":extra.tolist(),"loads":state["loads"].tolist(),
                        "local_fraction":float(state["local"].sum()/demand.sum())}
                reference=None
                order=list(configs)
                rng=random.Random(7317+layer*100+group_idx)
                for rep in range(-args.warmup,args.reps):
                    rng.shuffle(order)
                    # Reference must exist for correctness checks, independent
                    # of randomized measured order.
                    if reference is None:
                        order=["home_single"]+[x for x in order if x!="home_single"]
                    for name in order:
                        c=configs[name]
                        torch.cuda.synchronize();dist.barrier()
                        ev=[torch.cuda.Event(enable_timing=True) for _ in range(7)]
                        ev[0].record()
                        with torch.cuda.stream(comm):
                            comm.wait_event(ev[0]);ev[1].record()
                            work=dist.all_gather_into_tensor(all_hidden,hidden,async_op=True)
                            work.wait();ev[2].record()
                        if name=="home_single":
                            local_out=torch.zeros_like(hidden)
                        else:
                            local_out=fused_experts(hidden,c["w1"],c["w2"],c["local_weights"],c["local_ids"],
                                expert_map=c["emap"],global_num_experts=128)
                        ev[3].record()
                        stream.wait_event(ev[2])
                        remote_out=fused_experts(all_hidden,c["w1"],c["w2"],
                            all_weights if name=="home_single" else c["remote_weights"],
                            all_ids if name=="home_single" else c["remote_ids"],
                            expert_map=c["emap"],global_num_experts=128)
                        ev[4].record()
                        combined=torch.empty_like(hidden)
                        dist.reduce_scatter_tensor(combined,remote_out)
                        ev[5].record()
                        output=combined+local_out
                        ev[6].record();ev[6].synchronize()
                        if reference is None:
                            reference=output.clone()
                        error=(output.float()-reference.float())
                        relative=float(error.norm()/reference.float().norm().clamp_min(1e-9))
                        assert relative<.025,(rank,layer,group_idx,modality,name,relative)
                        if rep>=0:
                            row={"rank":rank,"layer":layer,"group":group_idx,"requests":group,
                                 "modality":modality,"policy":name,"repeat":rep,"tokens":len(sel),
                                 "all_tokens":sum(map(len,selections)),"padded_tokens":cap*4,
                                 "dispatch_ms":ev[1].elapsed_time(ev[2]),
                                 "local_ms":ev[0].elapsed_time(ev[3]),
                                 "remote_after_local_ms":ev[3].elapsed_time(ev[4]),
                                 "combine_ms":ev[4].elapsed_time(ev[5]),"whole_ms":ev[0].elapsed_time(ev[6]),
                                 "replication_cold_ms":c["copy_ms"],"replicated_experts":c["extra"],
                                 "rank_loads":c["loads"],"local_fraction":c["local_fraction"],
                                 "relative_l2":relative,"max_abs":float(error.abs().max()),
                                 "python_planner_ms_not_official":cpu_plan_ms,
                                 "scope":"FIXED_REAL_INPUT_MECHANISM_ORACLE_NOT_SERVING"}
                            results.append(row)
                            with (args.out/f"rank{rank}.jsonl").open("a") as f:
                                f.write(json.dumps(row)+"\n")
                print(json.dumps({"rank":rank,"layer":layer,"group":group_idx,"modality":modality,"status":"PASS"}),flush=True)
                del configs,reference,all_hidden,all_weights,hidden
                gc.collect();torch.cuda.empty_cache()
        del peers1,peers2,hdl1,hdl2,base1,base2
        gc.collect();torch.cuda.empty_cache()
    (args.out/f"completed_{rank}.json").write_text(json.dumps({"started_utc":started,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"physical_gpu":physical_gpu(rank),
        "observations":len(results),"scope":"PAPER_MECHANISM_DIAGNOSTIC_NOT_OFFICIAL_SERVING"},indent=2))
    dist.destroy_process_group()


if __name__=="__main__":main()
