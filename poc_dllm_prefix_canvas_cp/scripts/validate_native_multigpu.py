"""Bounded native TP4 sanity, before the discovery cohort is executed."""
import argparse
import os
import signal
import socket
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_dllm_prefix_canvas_cp.contract import *
from poc_dllm_prefix_canvas_cp.runtime import setup


def worker(rank,port,args):
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    model,runner,dllm,tokenizer=setup(rank,port,args)
    prompt='<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Write a Python function that adds two integers.<|role_end|><role>ASSISTANT</role>'
    ids=tokenizer(prompt,return_tensors='pt').input_ids.to(rank)
    results=[]
    for repeat in range(3):
        dist.barrier();torch.cuda.synchronize();start=time.perf_counter();nfe=dllm.num_forwards
        output=dllm.generate(ids,gen_length=64,block_length=32)
        torch.cuda.synchronize()
        results.append(dict(repeat=repeat,wall_ms=(time.perf_counter()-start)*1000,
            NFE=dllm.num_forwards-nfe,output_ids=output[0].cpu().tolist()))
    assert all(r['output_ids']==results[0]['output_ids'] and r['NFE']==results[0]['NFE'] for r in results)
    gathered=[None]*4;dist.all_gather_object(gathered,results)
    assert all(v[0]['output_ids']==results[0]['output_ids'] for v in gathered)
    write(Path(args.output)/f'sanity.rank{rank}.json',dict(parity=True,results=results))
    if rank==0:print('Native',args.backend,'sanity PASS',[(r['NFE'],round(r['wall_ms'],1)) for r in results],flush=True)
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--backend',choices=['tp4','ep4'],default='tp4')
    p.add_argument('--graphs',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,a),nprocs=4,join=True)
    write(Path(a.output)/'status.json',dict(success=True,GPUS=GPUS,backend=a.backend))


if __name__=='__main__':main()
