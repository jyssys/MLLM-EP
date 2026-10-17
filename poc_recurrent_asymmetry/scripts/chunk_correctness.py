"""Bounded full-cache checks for native serial chunking, not a pipeline engine."""
import gc
import os
import torch
from poc_recurrent_asymmetry.contract import *
from poc_async_recurrent_pp.scripts.baseline import load


@torch.inference_mode()
def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    limits=dict(logits_max_abs=.5,logits_mean_abs=.05,KV_relative_L2=.02)
    write(output/'preflight.json',dict(limits=limits,context=1024,chunks=[128,256,512,1024],
        requests=2,cached_steps=3,scope='Same limits as the earlier native diagnostic. No tolerance relaxation.'))
    model,_=load();rows=[]
    def call(ids,past=None):
        return model(ids,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
    def snapshot(value):
        return dict(logits=value.logits.cpu().float(),
            keys=[t.cpu() for t in value.past_key_values.key_cache],
            values=[t.cpu() for t in value.past_key_values.value_cache])
    for request in read(ROOT/'metrics/frozen_workload.json')['requests'][:2]:
        ids=torch.tensor([request['prompts']['1024']],device='cuda')
        token=torch.tensor([[13]],device='cuda')
        value=call(ids);reference=[snapshot(value)]
        for _ in range(3):
            value=call(token,value.past_key_values);reference.append(snapshot(value))
        del value;gc.collect();torch.cuda.empty_cache()
        for chunk in [128,256,512,1024]:
            past=None
            for offset in range(0,1024,chunk):
                value=call(ids[:,offset:offset+chunk],past);past=value.past_key_values
            for step in range(4):
                if step:value=call(token,value.past_key_values)
                observed=snapshot(value);gold=reference[step]
                delta=(observed['logits']-gold['logits']).abs()
                square=0.;denominator=0.;bitwise=True;worst=0.
                for kind in ['keys','values']:
                    assert len(observed[kind])==96
                    for a,b in zip(observed[kind],gold[kind]):
                        assert a.shape==b.shape
                        d=(a.float()-b.float())
                        square+=float(d.square().sum());denominator+=float(b.float().square().sum())
                        worst=max(worst,float(d.abs().max()));bitwise &= torch.equal(a,b)
                relative=(square/max(denominator,1e-30))**.5
                row=dict(request=request['id'],chunk=chunk,cached_step=step,
                    logits_max_abs=float(delta.max()),logits_mean_abs=float(delta.mean()),
                    KV_relative_L2=relative,KV_max_abs=worst,KV_bitwise=bitwise,
                    top1_equal=bool(observed['logits'].argmax()==gold['logits'].argmax()),
                    pass_numerical=float(delta.max())<=limits['logits_max_abs']
                        and float(delta.mean())<=limits['logits_mean_abs'] and relative<=limits['KV_relative_L2'])
                rows.append(row)
            write(output/'checks.json',rows)
            del value,past;gc.collect();torch.cuda.empty_cache()
            print('CHUNK_KV',request['id'],chunk,flush=True)
    write(output/'status.json',dict(success=True,cases=len(rows),limits=limits,
        all_numerical=all(r['pass_numerical'] for r in rows),all_top1=all(r['top1_equal'] for r in rows),
        scope='Two calibration requests, full 96-slot KV and three forced continuations. No free-running pipeline or quality claim.'))


if __name__=='__main__':main()
