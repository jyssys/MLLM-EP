"""Bounded calibration logit-margin diagnostic. It cannot select or tune a policy."""
import copy
import os
import torch
import numpy as np
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scripts.baseline import load,token_ids
from poc_async_recurrent_pp.emulator import StageEmulator


@torch.inference_mode()
def main():
    model,tokenizer=load()
    data=read(ROOT/'outputs/manifests/data.json')
    coef=read(ROOT/'outputs/manifests/predictor.json')
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    records=[]
    for row in data['calibration'][:2]:
        ids=token_ids(tokenizer,row,data['prompt_suffix'])
        value=model(torch.tensor([ids],device='cuda'),use_cache=True,exit_at_step=3,logits_to_keep=1)
        prefix=copy.deepcopy(value.past_key_values)
        teacher=[];baseline=[]
        token=int(value.logits[0,-1].argmax())
        for step in range(16):
            if token==tokenizer.eos_token_id:
                break
            teacher.append(token)
            value=model(torch.tensor([[token]],device='cuda'),past_key_values=value.past_key_values,
                        use_cache=True,exit_at_step=3,logits_to_keep=1)
            baseline.append(value.logits[0,-1].float().clone())
            token=int(value.logits[0,-1].argmax())
        for arm in ['stale','delta','w0']:
            hook=StageEmulator(model,arm,coef)
            past=copy.deepcopy(prefix)
            try:
                for step,token in enumerate(teacher):
                    value=model(torch.tensor([[token]],device='cuda'),past_key_values=past,
                                use_cache=True,exit_at_step=3,logits_to_keep=1)
                    logits=value.logits[0,-1].float()
                    a=baseline[step].topk(2).values
                    b=logits.topk(2).values
                    p=baseline[step].log_softmax(-1);q=logits.log_softmax(-1)
                    records.append(dict(id=row['id'],step=step,arm=arm,
                        exact_top1_top2_logit_margin=float(a[0]-a[1]),
                        candidate_top1_top2_logit_margin=float(b[0]-b[1]),
                        KL_exact_candidate_nats=float((p.exp()*(p-q)).sum()),
                        top1_agreement=bool(logits.argmax()==baseline[step].argmax()),
                        finite=bool(torch.isfinite(logits).all())))
            finally:
                hook.close()
    write(output/'margin_samples.json',records)
    summary=[dict(arm=arm,n_positions=sum(r['arm']==arm for r in records),
        exact_margin_mean=float(np.mean([r['exact_top1_top2_logit_margin'] for r in records if r['arm']==arm])),
        candidate_margin_mean=float(np.mean([r['candidate_top1_top2_logit_margin'] for r in records if r['arm']==arm])),
        all_finite=all(r['finite'] for r in records if r['arm']==arm)) for arm in ['stale','delta','w0']]
    write(output/'summary.json',dict(rows=summary,independent_calibration_requests=2,
        scope='Post-screen calibration diagnostic. Teacher forcing only. No fitting, policy selection, or task-accuracy claim.',
        predictor_sha256=sha(ROOT/'outputs/manifests/predictor.json')))
    print(summary,flush=True)


if __name__=='__main__':
    main()
