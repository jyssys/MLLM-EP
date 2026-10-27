"""Collect real routes on disjoint prompts. Sampling does not depend on routing."""
import gzip
import json
import torch
from .common import write
from .traces import RouterTrace
from .workloads import inputs

def collect(a,target,config):
    count=0;cases=[]
    with gzip.open(target/'router_corpus.jsonl.gz','wt') as output:
        for split in ['calibration','heldout']:
            rows_per_domain=8 if split=='calibration' else 16
            for b in [1,2,4,8,16]:
                offsets=range(max(1,rows_per_domain//max(1,b//4)))
                domains=range(0,4,b) if b<4 else [0]
                for length in [128,512,2048]:
                    for offset in offsets:
                        for domain_offset in domains:
                            ids,requests=inputs(a,b,length,split,offset*max(1,b//4),domain_offset)
                            trace=RouterTrace(a)
                            trace.request_ids=[r['id'] for r in requests]
                            trace.domain=[r['domain'] for r in requests]
                            result=a.forward(ids)
                            # A whole-vocabulary finite check creates large temporary tensors.
                            # Check every logit in bounded views instead. Model execution is unchanged.
                            flat=result.reshape(-1,result.shape[-1])
                            for block in flat.split(16):assert torch.isfinite(block).all()
                            del flat,block
                            del result
                            trace.close()
                            for record in trace.records:
                                record.update(split=split,model=a.name)
                                output.write(json.dumps(record)+'\n');count+=1
                            cases.append(dict(split=split,batch=b,context=length,requests=trace.request_ids))
                            output.flush()
                    print('ROUTES',a.name,split,b,length,count,flush=True)
    write(target/'route_manifest.json',dict(model=a.name,records=count,cases=cases,
        token_sample='Up to 64 uniform positions per sequence. Same positions at each pass.',
        scope='LIVE_NATIVE_ROUTES. Full native model. No EP cost inferred from these runs.'))
