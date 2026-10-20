"""Isolate shared graph scratch from cross-device transport."""
import json
import argparse
import torch
from .contract import ROOT,write
from .model_adapter import Part
from .reference import golden,errors


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='v3_failed');args=parser.parse_args()
    torch.set_num_threads(1)
    requests=json.loads((ROOT/'workload.json').read_text())['groups']['16'][:4]
    ref=golden(requests,4,4)
    part=Part(0,'loop_pp4');part.configure(requests,8,True);part.reset()
    out={r['id']:[] for r in requests}
    stream=torch.cuda.Stream(device=0)
    for step in range(4):
        for r in requests:
            rid=r['id']
            tokens=r['input_ids'] if step==0 else [ref[rid]['ids'][step-1]]
            with torch.cuda.stream(stream):
                h=part.embedding(torch.tensor([tokens],device='cuda:0'))
                for loop in range(4):
                    h=part.stage(h,rid,step,0 if step==0 else 16+step-1,loop,0,24)
                logits=part.head(h[:,-1:,:])
                token=int(logits.argmax(-1))
            out[rid].append(token)
            print(step,rid,token,ref[rid]['ids'][step],flush=True)
    write(ROOT/f'failures/interleave_probe_{args.label}.json',dict(observed=out,expected={k:v['ids'] for k,v in ref.items()}))
    assert all(out[k]==ref[k]['ids'] for k in out)


if __name__=='__main__':main()
