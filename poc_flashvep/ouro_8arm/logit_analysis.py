"""Distribution errors on teacher-aligned synthetic logits, not task accuracy."""
import csv
import torch
from .contract import *
from .numerical_trace import errors


def main():
    torch.set_num_threads(1);rows=[]
    for meta in read(ROOT/'numerical/hf_manifest.json'):
        label=meta['label'];values={}
        for name in ['HF','Unsharded','Custom_TP4','vLLM_TP1','vLLM_TP4','FP32_math']:
            path=ROOT/'.tensor_cache'/f'{name}_{label}.pt'
            if path.exists():values[name]=torch.load(path,map_location='cpu',weights_only=False)['logits']
        pairs=[('HF','Unsharded'),('HF','Custom_TP4'),('HF','vLLM_TP1'),('HF','vLLM_TP4'),('vLLM_TP1','vLLM_TP4')]
        pairs += [('FP32_math',candidate) for candidate in ['HF','Unsharded','Custom_TP4','vLLM_TP1','vLLM_TP4']]
        for gold,candidate in pairs:
            if gold not in values or candidate not in values:continue
            for step,(a,b) in enumerate(zip(values[gold],values[candidate])):
                a=a.reshape(-1).float();b=b.reshape(-1).float();lp=a.log_softmax(-1);lq=b.log_softmax(-1)
                p=lp.exp();q=lq.exp();lm=torch.logaddexp(lp,lq)-__import__('math').log(2)
                top=a.topk(2).values
                rows.append(dict(pair=gold+' vs '+candidate,label=label,P=meta['P'],step_zero_based=step,
                    **errors(b,a),reference_RMS=float(a.square().mean().sqrt()),KL_reference_to_candidate=float((p*(lp-lq)).sum()),
                    JS_divergence=float((p*(lp-lm)).sum()/2+(q*(lq-lm)).sum()/2),
                    argmax_equal=int(a.argmax())==int(b.argmax()),reference_top2_margin=float(top[0]-top[1]),
                    top5_overlap=len(set(a.topk(5).indices.tolist())&set(b.topk(5).indices.tolist()))))
    with (ROOT/'logit_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('ALIGNED_LOGIT_METRICS',len(rows),flush=True)


if __name__=='__main__':main()
