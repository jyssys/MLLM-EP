"""Real four-GPU dispatcher/owner-GEMM diagnostic, not a model benchmark.

Each process holds only its 64 experts. The reference uses the same local
expert shards and an all-reduce after the real DeepEP path completes. No
single-GPU routing projection is presented as physical EP evidence.
"""
import argparse
import datetime
import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from poc_multibd_moe_ep.contract import *


def worker(rank, port, args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == MASK
    torch.cuda.set_device(rank)
    uuid = str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')
    assert uuid == UUIDS[rank]
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,
                            world_size=4,timeout=datetime.timedelta(seconds=120))
    sys.path.insert(0,str(UPSTREAM))
    from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher
    from diffulex_kernel import fused_expert_packed
    from diffulex_kernel.python.fused_moe_triton import fused_moe
    if args.repair_local_ids:
        from poc_multibd_moe_ep.bounded_bridge import repair_normal_owner_local_ids
        repair_normal_owner_local_ids()
    device = torch.device('cuda',rank)
    gen = torch.Generator(device=device).manual_seed(3407)
    hidden = torch.randn((128,2048),generator=gen,device=device,dtype=torch.bfloat16)*.1
    scores = torch.randn((128,256),generator=gen,device=device)
    ids = scores.topk(8,dim=-1).indices.to(torch.int32)
    weights = torch.softmax(scores.topk(8,dim=-1).values,dim=-1)
    owned = torch.Generator(device=device).manual_seed(1000+rank)
    w13 = torch.randn((64,1024,2048),generator=owned,device=device,dtype=torch.bfloat16)*.02
    w2 = torch.randn((64,2048,512),generator=owned,device=device,dtype=torch.bfloat16)*.02
    dispatcher = DeepEPDispatcher(ep_group=dist.group.WORLD,ep_size=4,
        num_local_experts=64,num_experts=256,top_k=8,hidden_size=2048,
        params_dtype=torch.bfloat16,deepep_mode='normal',async_finish=False)
    local_h, local_ids, local_w = (x[rank::4].contiguous() for x in (hidden,ids,weights))
    dispatched = dispatcher.dispatch(local_h,local_ids,local_w)
    meta = dispatched.metadata
    valid_native = meta.native_recv_topk_ids[meta.native_recv_topk_ids>=0]
    expected_paths = int(((ids//64)==rank).sum().item())
    expected_rows = int(((ids//64)==rank).any(-1).sum().item())
    info = dict(rank=rank, physical_gpu=GPUS[rank], uuid=uuid,
        owner_ids=list(range(rank*64,(rank+1)*64)), w13=list(w13.shape),w2=list(w2.shape),
        expert_bytes_per_rank=w13.numel()*2+w2.numel()*2,
        native_received_ids_min=int(valid_native.min().item()),
        native_received_ids_max=int(valid_native.max().item()),
        expected_paths=expected_paths, expected_destination_dedup_rows=expected_rows,
        actual_native_rows=meta.native_recv_num_tokens, actual_expert_slots=meta.total_recv_slots,
        backend='deep_ep.Buffer.dispatch/combine normal',
        source_layout='strided row_id % 4',repair_local_ids=args.repair_local_ids,
        lane='DIAGNOSTIC_SYNTHETIC_REAL_PHYSICAL_EP4_NOT_FULL_MODEL')
    if meta.total_recv_slots:
        branch = fused_expert_packed(hidden_states=dispatched.recv_hidden_states,
            w13=w13,w2=w2,execution_metadata=meta.to_expert_execution_metadata(),hidden_act='silu')
        branch.mul_(meta.recv_weights.to(branch.dtype).unsqueeze(-1))
    else:
        branch = torch.empty((0,2048),device=device,dtype=torch.bfloat16)
    output = dispatcher.combine(branch,meta)
    torch.cuda.synchronize()
    # All ranks compute only owned experts in the comparison reference too.
    reference = fused_moe(hidden_states=hidden,w13=w13,w2=w2,
        topk_ids=ids,topk_weights=weights.to(torch.bfloat16),
        local_expert_start=rank*64,hidden_act='silu')
    dist.all_reduce(reference)
    ref = reference[rank::4]
    err = output.float()-ref.float()
    info.update(rel_l2=float(err.norm()/ref.float().norm().clamp_min(1e-12)),
        max_scaled_abs=float(err.abs().max()/ref.float().abs().max().clamp_min(1e-12)))
    info['path_accounting_pass'] = (meta.total_recv_slots==expected_paths and
                                  meta.native_recv_num_tokens==expected_rows)
    info['correctness_pass'] = (info['rel_l2']<=.02 and info['max_scaled_abs']<=.04
                                and info['path_accounting_pass'])
    write(Path(args.output)/f'rank{rank}.json',info)
    summaries=[None]*4
    dist.all_gather_object(summaries,info)
    if rank==0:
        write(Path(args.output)/'summary.json',dict(ranks=summaries,
            pass_all=all(r['correctness_pass'] for r in summaries),
            full_model_ep_validated=False))
        print(json.dumps(summaries),flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not info['correctness_pass']:
        raise RuntimeError(f'Physical dispatcher correctness failed on rank{rank}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--repair-local-ids',action='store_true');args=p.parse_args()
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,args),nprocs=4,join=True)


if __name__=='__main__':
    main()
