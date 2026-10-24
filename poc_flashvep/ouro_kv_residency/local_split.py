"""Generic Flash-Decoding-style segmentation with existing FA4 and exact merge."""
import json
import torch
from .common import ROOT,FIXTURES,write
from . import distributed as ops

def split_cache(cache,length,batch,splits):
    k,v,original_table,original_lengths=cache
    pages=original_table.shape[1];tables=[];lengths=[]
    for s in range(splits):
        lo=pages*s//splits;hi=pages*(s+1)//splits
        tables.append(original_table[:,lo:hi])
        lengths.append(torch.full((batch,),max(0,min(length-lo*16,(hi-lo)*16)),device='cuda',dtype=torch.int32))
    width=max(t.shape[1] for t in tables)
    padded=[torch.nn.functional.pad(t,(0,width-t.shape[1])) for t in tables]
    return k,v,torch.cat(padded),torch.cat(lengths)

def run(job,target):
    rows=[]
    for context in job.get('contexts',[2048,8192,32768,65408]):
        packet=torch.load(FIXTURES/f'P{context}_d1_l12.pt',map_location='cpu',weights_only=True)
        for batch in job.get('batches',[1,8]):
            for pressure in (False,True):
                ops.PRESSURE=torch.zeros(32*1024*1024,device='cuda') if pressure else None
                q=packet['q'].repeat(batch,1,1).to('cuda')
                base_cache=ops.paged(packet['k'],packet['v'],batch)
                native=lambda:ops.attention(q,base_cache,return_lse=False)[0].squeeze(1)
                g,reference=ops.graph(native);reference=reference.clone()
                baseline=ops.bench(g.replay,job.get('repeats',25));del g,native
                for splits in (2,4,8,16):
                    cache=split_cache(base_cache,packet['k'].shape[0],batch,splits)
                    def segmented():
                        qs=q.repeat(splits,1,1)
                        out,lse=ops.attention(qs,cache)
                        weights=torch.softmax(lse.reshape(splits,batch,16),dim=0)
                        return (out.squeeze(1).view(splits,batch,16,128).float()*weights.unsqueeze(-1)).sum(0).to(torch.bfloat16)
                    g,result=ops.graph(segmented);numeric=ops.metrics(reference,result)
                    assert numeric['relative_L2']<=.03 and numeric['absolute_max']<=.125 and numeric['finite'],numeric
                    measurements=ops.bench(g.replay,job.get('repeats',25))
                    rows.append(dict(L=context+1,B=batch,S=splits,cache_pressure=pressure,baseline=baseline,
                        segmented=measurements,numerical=numeric,evidence='LIVE_MEASURED_OPERATOR',
                        mechanism='Generic single-GPU context segmentation. One FA4 batched call plus stable merge. No new kernel or loop-specific policy.'))
                    write(target/'local_segmentation.json',rows)
                    del g,result,cache,segmented
                del q,reference,base_cache
                ops.PRESSURE=None;torch.cuda.empty_cache()
    return dict(rows=rows)
