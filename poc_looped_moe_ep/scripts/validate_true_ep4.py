"""Tiny authentic pretrained reference→sharded-EP4 gate, not a random probe."""
import argparse
import datetime
import gzip
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_looped_moe_ep.contract import *
from poc_looped_moe_ep.runtime import load_model,PhysicalEP,pools


def worker(rank,port,output):
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4,
                            timeout=datetime.timedelta(seconds=120))
    model,tokenizer,snapshot=load_model(torch.device('cuda',rank))
    ids=tokenizer('The history of science is a story of',return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        reference=model(input_ids=ids).logits.detach()
        repeated=model(input_ids=ids).logits.detach()
        assert torch.equal(reference,repeated),'Native reference is not deterministic'
        ep=PhysicalEP(model,rank);ep.mode='trace'
        actual=model(input_ids=ids).logits.detach();records=ep.resolved()
        delta=actual.float()-reference.float()
        rel=float(delta.norm()/reference.float().norm().clamp_min(1e-9))
        scaled=float(delta.abs().max()/reference.float().abs().max().clamp_min(1e-9))
        info=dict(rank=rank,physical_gpu=GPUS[rank],uuid=UUIDS[rank],checkpoint=MODEL,
            revision=REVISION,native_H=16,ownership=ep.ownership,
            topology='TP1 replicated dense / source-SP4 / physical EP4 / DP1 / PP1',
            reference_repeat_exact=True,rel_l2=rel,max_scaled_abs=scaled,
            next_token_equal=bool(torch.equal(actual[:,-1].argmax(-1),reference[:,-1].argmax(-1))),
            reference_next_token=int(reference[0,-1].argmax()),
            actual_next_token=int(actual[0,-1].argmax()),
            calibrated_tolerance=dict(relative_L2=.02,scaled_max_abs=.04),
            native_cache_policy='full-prefix recomputation; no KV cache',
            small_model_only=True)
        info['pass_all']=rel<=.02 and scaled<=.04 and info['next_token_equal']
        write(Path(output)/f'rank{rank}.json',info)
        with gzip.open(Path(output)/f'trace.rank{rank}.json.gz','wt') as out:__import__('json').dump(records,out)
        assert info['pass_all'],info
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,a.output),nprocs=4,join=True)


if __name__=='__main__':main()
