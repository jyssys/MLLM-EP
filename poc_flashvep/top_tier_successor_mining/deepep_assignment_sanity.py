"""Distributed compatibility test for paper-baseline routing outputs.

Identity experts make the exact combine answer known. This tests duplicate expert
IDs (SERE) and -1 sentinels (MoDES), not method performance or scientific failure.
"""
import argparse
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices, physical_gpu

import torch
import torch.distributed as dist
import deep_ep


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    allowed_devices()
    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group("nccl",device_id=torch.device(f"cuda:{rank}"))
    assert dist.get_world_size()==4
    started=datetime.now(timezone.utc).isoformat()
    dc=deep_ep.Buffer.get_dispatch_config(4)
    cc=deep_ep.Buffer.get_combine_config(4)
    size=max(dc.get_nvl_buffer_size_hint(4096,4),cc.get_nvl_buffer_size_hint(4096,4))
    buffer=deep_ep.Buffer(dist.group.WORLD,size,0)
    streams=[torch.cuda.current_stream(),torch.cuda.Stream()]
    rows=[]
    for stream_id,stream in enumerate(streams):
        for m in [1,16,64]:
            for variant in ["unique","duplicates","sentinel","duplicates_and_empty_source"]:
                for repeat in range(3):
                    with torch.cuda.stream(stream):
                        torch.manual_seed(1731+rank+m)
                        hidden=torch.randn(m,2048,device="cuda",dtype=torch.bfloat16)
                        ids=torch.rand(m,128,device="cuda").topk(8).indices.to(deep_ep.topk_idx_t)
                        weights=torch.rand(m,8,device="cuda",dtype=torch.float32).softmax(-1)
                        if variant.startswith("duplicates"):
                            ids=ids[:,:2].repeat(1,4).contiguous()
                        if variant=="sentinel":
                            ids[:,3:]=-1
                            weights[:,3:]=0
                        if variant=="duplicates_and_empty_source" and rank%2:
                            ids.fill_(-1)
                            weights.zero_()
                        layout=buffer.get_dispatch_layout(ids,128,async_finish=True)
                        nr,nd,ne,inc,event=layout
                        rx,ri,rw,counts,handle,event=buffer.dispatch(
                            hidden,num_tokens_per_rank=nr,num_tokens_per_rdma_rank=nd,
                            num_tokens_per_expert=ne,is_token_in_rank=inc,
                            topk_idx=ids,topk_weights=weights,config=dc,
                            previous_event=event,async_finish=True,allocate_on_comm_stream=False)
                        event.current_stream_wait()
                        value=(rx.float()*(rw*(ri>=0)).sum(-1,keepdim=True)).to(torch.bfloat16)
                        out,_,event=buffer.combine(value,handle,config=cc,
                            previous_event=buffer.capture(),async_finish=True,allocate_on_comm_stream=False)
                        event.current_stream_wait()
                        expected=(hidden.float()*weights.sum(-1,keepdim=True)).to(torch.bfloat16)
                        torch.cuda.current_stream().synchronize()
                        error=(out.float()-expected.float()).abs()
                        relative=(out.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-9)
                        assert float(relative)<.01,(rank,m,variant,float(relative))
                        rows.append({"rank":rank,"M":m,"variant":variant,"repeat":repeat,
                                     "stream":stream_id,"max_abs":float(error.max()),"relative_l2":float(relative),
                                     "received_tokens":rx.shape[0],"received_assignments":sum(counts)})
                    dist.barrier()
                print(json.dumps({"rank":rank,"M":m,"variant":variant,"stream":stream_id,"status":"PASS"}),flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f"rank{rank}.json").write_text(json.dumps({"started_utc":started,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"physical_gpu":physical_gpu(rank),
        "deepep_source":deep_ep.__file__,"nvl_buffer_bytes":size,"tests":rows},indent=2))
    dist.destroy_process_group()


if __name__=="__main__":main()
