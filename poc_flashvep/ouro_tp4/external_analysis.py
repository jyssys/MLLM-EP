"""Compare native vLLM numerical observations with the unchanged HF reference."""
import torch
from poc_flashvep.ouro_request_pp.reference import errors
from .contract import *


def main():
    torch.set_num_threads(1);results=[]
    for P,O in [(128,8),(512,32),(2048,32)]:
        gold=next(iter(torch.load(ROOT/f'.tensor_cache/native_P{P}_O{O}.pt',weights_only=False).values()))
        for rank in range(4):
            data=torch.load(ROOT/f'.tensor_cache/external_P{P}_O{O}_rank{rank}.pt',weights_only=False)
            logit=[errors(r['logits'].reshape(-1),gold['logits'][step]) for step,r in enumerate(data['rows'])]
            hidden=[errors(r['hidden'].view(1,1,-1),gold['hidden'][step]) for step,r in enumerate(data['rows'])]
            ids=[int(r['logits'].argmax()) for r in data['rows']]
            num=den=0.;maximum=0.
            for slot,kv in data['KV'].items():
                k,v=gold['cache'][slot]
                reference=torch.stack([torch.cat([k[0,rank*4:(rank+1)*4,p],v[0,rank*4:(rank+1)*4,p]],dim=-1) for p in data['positions']])
                a,b=kv.float(),reference.float();delta=a-b;num+=float(delta.square().sum());den+=float(b.square().sum());maximum=max(maximum,float(delta.abs().max()))
            kv_l2=(num/max(den,1e-30))**.5
            stable=[i for i,margin in enumerate(gold['margins']) if margin>=TOL['stable_native_margin']]
            row=dict(P=P,O=O,rank=rank,logit_errors=logit,hidden_errors=hidden,
                KV_sample_relative_L2=kv_l2,KV_sample_max_abs=maximum,KV_slots=len(data['KV']),positions=data['positions'],
                native_ids=gold['ids'],observed_argmax_ids=ids,token_exact=ids==gold['ids'],
                stable_mismatches=[i for i in stable if ids[i]!=gold['ids'][i]],
                max_logit_abs=max(e['max_abs'] for e in logit),max_logit_mean=max(e['mean_abs'] for e in logit),
                max_hidden_relative_L2=max(e['relative_L2'] for e in hidden),
                scope='Teacher-aligned numerical observer. KV compares sampled positions, not every stored element.')
            row['PASS']=(row['max_logit_abs']<=TOL['logit_max_abs'] and row['max_logit_mean']<=TOL['logit_mean_abs']
                and row['max_hidden_relative_L2']<=TOL['hidden_relative_L2'] and kv_l2<=TOL['KV_relative_L2'] and not row['stable_mismatches'])
            results.append(row)
    write(ROOT/'external_numerical_comparison.json',results)
    print('EXTERNAL_NUMERICAL',sum(r['PASS'] for r in results),'/',len(results),flush=True)


if __name__=='__main__':main()
