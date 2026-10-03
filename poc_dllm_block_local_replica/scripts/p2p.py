"""Physical CUDA P2P: exact6MiB expert shape and bulk transfers, no speedup projection."""
import argparse
import json
import os
import subprocess
import time
import torch
from poc_dllm_block_local_replica.contract import MASK, GPUS, UUIDS, EXPERT_BYTES, new_json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == MASK
    assert torch.cuda.device_count() == 4
    props = []
    for r in range(4):
        prop=torch.cuda.get_device_properties(r)
        assert str(prop.uuid).removeprefix('GPU-') == UUIDS[r]
        props.append(dict(logical=r,physical=GPUS[r],uuid=str(prop.uuid),name=prop.name,
                          total_memory=prop.total_memory,free_memory=torch.cuda.mem_get_info(r)[0]))
    new_json(out/'clean_p2p_start.json',dict(gpus=props,topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)))
    rows=[]
    for size in (EXPERT_BYTES,96*1024**2,384*1024**2):
        tensors=[torch.ones(size//2,device=r,dtype=torch.bfloat16) for r in range(4)]
        targets=[torch.zeros_like(t) for t in tensors]
        for src in range(4):
            for dst in range(4):
                if src==dst: continue
                assert torch.cuda.can_device_access_peer(src,dst)
                torch.cuda.set_device(dst)
                for _ in range(5): targets[dst].copy_(tensors[src],non_blocking=True)
                torch.cuda.synchronize(dst)
                pairs=[]
                for _ in range(30):
                    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                    s.record();targets[dst].copy_(tensors[src],non_blocking=True);e.record();pairs.append((s,e))
                torch.cuda.synchronize(dst)
                ms=[s.elapsed_time(e) for s,e in pairs]
                assert torch.equal(targets[dst],tensors[dst])
                rows.append(dict(src=src,dst=dst,physical_src=GPUS[src],physical_dst=GPUS[dst],bytes=size,
                                 mode='contiguous_P2P',samples_ms=ms))
                if size==EXPERT_BYTES:
                    # Loaded expert has w13=4MiB plus w2=2MiB, not one tensor.
                    n13=4*1024**2//2;pairs=[]
                    for _ in range(30):
                        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                        s.record();targets[dst][:n13].copy_(tensors[src][:n13],non_blocking=True)
                        targets[dst][n13:].copy_(tensors[src][n13:],non_blocking=True);e.record();pairs.append((s,e))
                    torch.cuda.synchronize(dst)
                    rows.append(dict(src=src,dst=dst,physical_src=GPUS[src],physical_dst=GPUS[dst],bytes=size,
                                     mode='actual_two_tensor_P2P',samples_ms=[s.elapsed_time(e) for s,e in pairs]))
        del tensors,targets
        torch.cuda.empty_cache()
    new_json(out/'p2p.json',dict(mask=MASK,rows=rows,warmup=5,repeats=30,torch=torch.__version__,cuda=torch.version.cuda,
        timing='destination-stream CUDA events; P2P copy physical; serial directed edges, not concurrent all-edge measurement',
        caution='P2P transfer BW is not DeepEP dispatch BW; copy estimates are not runtime benefit estimates'))
    new_json(out/'status.json',dict(ok=True,status='COMPLETE'))
    print('P2P_COMPLETE',len(rows),flush=True)


if __name__=='__main__': main()
