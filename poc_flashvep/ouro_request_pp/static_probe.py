"""Minimized GPU regression: one request, two forwards, no transport or worker."""
import argparse
import json
import torch
from .contract import ROOT,write
from .model_adapter import Part
from .reference import golden,errors


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--version',choices=['padded','exact_length'],default='padded');p.add_argument('--request-index',type=int,default=0);p.add_argument('--steps',type=int,default=2);args=p.parse_args()
    torch.set_num_threads(1)
    r=json.loads((ROOT/'workload.json').read_text())['groups']['16'][args.request_index:args.request_index+1]
    ref=golden(r,4,args.steps)[r[0]['id']]
    part=Part(0,'loop_pp4')
    part.graph_version=args.version
    rows=[]
    for graph in [False,True]:
        part.configure(r,8,True);part.reset();part.graphs.clear()
        part.capture_enabled=graph
        stream=torch.cuda.Stream(device=0)
        ids=[]
        with torch.cuda.stream(stream):
            for step in range(args.steps):
                tokens=r[0]['input_ids'] if step==0 else [ref['ids'][step-1]]
                h=part.embedding(torch.tensor([tokens],device='cuda:0'))
                for loop in range(4):
                    h=part.stage(h,r[0]['id'],step,0 if step==0 else 16+step-1,loop,0,24)
                logits=part.head(h[:,-1:,:]);ids.append(int(logits.argmax(-1)))
        torch.cuda.synchronize()
        row=dict(graph=graph,observed=ids,reference=ref['ids'],logit_error=errors(logits.cpu().float(),ref['logits']))
        rows.append(row)
        print(row,flush=True)
    write(ROOT/f'failures/static_probe_{args.label}.json',rows)
    assert all(x['observed']==x['reference'] for x in rows)


if __name__=='__main__':main()
