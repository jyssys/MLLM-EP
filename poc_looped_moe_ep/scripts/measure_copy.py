"""Reuses prior P2P methodology, with Foil's three actual 3-MiB tensor sizes."""
import argparse
import os
import torch
from poc_looped_moe_ep.contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK and torch.cuda.device_count()==4
    for r in range(4):
        assert str(torch.cuda.get_device_properties(r).uuid).removeprefix('GPU-')==UUIDS[r]
    rows=[]
    for size in [9*1024**2,96*1024**2,384*1024**2]:
        sources=[torch.ones(size//2,device=r,dtype=torch.bfloat16) for r in range(4)]
        targets=[torch.zeros_like(x) for x in sources]
        for src in range(4):
            for dst in range(4):
                if src==dst:continue
                assert torch.cuda.can_device_access_peer(src,dst)
                torch.cuda.set_device(dst)
                modes=['contiguous']+(['actual_3_tensors'] if size==9*1024**2 else [])
                for mode in modes:
                    def copy():
                        if mode=='contiguous':targets[dst].copy_(sources[src],non_blocking=True)
                        else:
                            chunk=3*1024**2//2
                            for i in range(3):
                                targets[dst][i*chunk:(i+1)*chunk].copy_(
                                    sources[src][i*chunk:(i+1)*chunk],non_blocking=True)
                    for _ in range(3):copy()
                    torch.cuda.synchronize(dst);events=[]
                    for _ in range(10):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record();copy();end.record();events.append((start,end))
                    torch.cuda.synchronize(dst)
                    assert torch.equal(targets[dst],sources[dst])
                    rows.append(dict(src=src,dst=dst,physical_src=GPUS[src],physical_dst=GPUS[dst],
                        bytes=size,mode=mode,samples_ms=[s.elapsed_time(e) for s,e in events]))
        del sources,targets;torch.cuda.empty_cache()
    write(Path(a.output)/'p2p.json',dict(mask=MASK,rows=rows,warmups=3,repeats=10,
        method='destination-stream CUDA events; serial isolated directed edges',
        reused='poc_dllm_block_local_replica/scripts/p2p.py methodology; native Foil expert shape corrected',
        caveat='warm buffers; not concurrent all-edge bandwidth, DeepEP bandwidth, reconfiguration cost or speedup'))


if __name__=='__main__':main()
