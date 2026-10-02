"""Physical 4-rank payload replay, not native prefill or model inference."""
import argparse
import collections
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import deep_ep

from poc_dispatch_combine_asymmetry.contract import MASK, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    if rank==0:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
    dist.barrier()
    group=dist.new_group(ranks=list(range(4)),backend='gloo')
    deep_ep.Buffer.set_num_sms(20)
    buffer=deep_ep.Buffer(group,num_nvl_bytes=1024<<20,num_rdma_bytes=0,num_qps_per_rank=1)
    captures=collections.defaultdict(dict)
    for path in args.captures.glob('*.pt'):
        # Trusted artifacts produced by this PoC, never external pickles.
        data=torch.load(path,map_location='cpu',weights_only=False)
        row=data['record']
        if 'hidden' in data and row['layer'] in (0,24,47):
            captures[(row['arm'],row['size'],row['layer'])][row['ep_rank']]=data
    pairs=[]
    for size in ('low','medium','high'):
        for layer in (0,24,47):
            for arm in ('visual','text'):
                data=captures[arm,size,layer]
                assert set(data)==set(range(4))
                n=data[0]['record']['prompt_tokens']
                # Restore source TP order, remove native SP padding, then
                # match token count across visual/text before resharding.
                matched=min(captures[a,size,layer][0]['record']['prompt_tokens'] for a in ('visual','text'))
                ids=torch.cat([data[r]['ids'] for r in (0,1)])[:n][:matched]
                weights=torch.cat([data[r]['weights'] for r in (0,1)])[:n][:matched]
                hidden=torch.cat([data[r]['hidden'] for r in (0,1)])[:n][:matched]
                start=rank*((matched+1)//2);stop=min((rank+1)*((matched+1)//2),matched)
                if rank<2:
                    local=[hidden[start:stop],ids[start:stop],weights[start:stop]]
                    if len(local[0])<(matched+1)//2:
                        # Retain the native sequence-parallel padding row.
                        local=[torch.cat([v,data[1][k][-1:]]) for v,k in zip(local,['hidden','ids','weights'])]
                else:
                    # Preserve installed idle-DP dummy calls/rows separately
                    # from useful global request accounting.
                    local=[data[rank][k] for k in ('hidden','ids','weights')]
                pairs.append((arm,size,layer,matched,*[v.contiguous().cuda() for v in local]))
    records=[]
    ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for rep in range(-3,args.repeats):
        ordered=pairs if rep%2==0 else pairs[::-1]
        for arm,size,layer,matched,x,ids,w in ordered:
            torch.cuda.synchronize();dist.barrier() # outside timer: all producers ready
            ev[0].record()
            layout=buffer.get_dispatch_layout(ids,128,async_finish=False,allocate_on_comm_stream=False)
            ev[1].record()
            rx,ri,rw,counts,handle,event=buffer.dispatch(x=x,num_tokens_per_rank=layout[0],
                num_tokens_per_rdma_rank=layout[1],num_tokens_per_expert=layout[2],is_token_in_rank=layout[3],
                topk_idx=ids,topk_weights=w,expert_alignment=1,async_finish=True,
                config=deep_ep.Buffer.get_dispatch_config(4),previous_event=buffer.capture(),allocate_on_comm_stream=False)
            event.current_stream_wait();ev[2].record()
            cx,_,event=buffer.combine(x=rx,handle=handle,topk_weights=None,async_finish=True,
                config=deep_ep.Buffer.get_combine_config(4),previous_event=buffer.capture(),allocate_on_comm_stream=False)
            event.current_stream_wait();ev[3].record()
            torch.cuda.synchronize()
            if rep>=0:
                records.append({'evidence':'ROUTE_CONTROLLED_TRUE_EP_REPLAY','arm':arm,'size':size,'layer':layer,
                    'repeat':rep,'ep_rank':rank,'source_rows':x.shape[0],'logical_prompt_tokens':matched,
                    'layout_ms':ev[0].elapsed_time(ev[1]),'dispatch_api_ms':ev[1].elapsed_time(ev[2]),
                    'dispatch_ms':ev[0].elapsed_time(ev[2]),'combine_ms':ev[2].elapsed_time(ev[3]),
                    'received_rows':rx.shape[0],'expert_histogram':counts,
                    'scope':'payload-only; native input hidden and routes; no expert kernel; all-rank ready barrier outside timing'})
    write(args.output/f'rank{rank}.json',records)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
