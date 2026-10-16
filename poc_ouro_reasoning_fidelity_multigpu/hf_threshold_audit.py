"""Inspect the released gate selector on one frozen calibration trajectory."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def audit(model,output):
    import torch
    row=next(r for r in read(ROOT/'results/fidelity_inputs.json') if r['id']=='60')
    counts=torch.zeros(4,device='cuda',dtype=torch.int64)
    snapshots=[]
    state={}
    index=0
    checkpoints={0,1,4,32,128,1024}
    def hook(module,args,values):
        hidden,gates=values[1],values[2]
        assert len(hidden)==4 and len(gates)==4
        last=[g[:,-1,:].squeeze(-1) for g in gates]
        remaining=torch.ones_like(last[0])
        pdf=[]
        for step,g in enumerate(last):
            probability=torch.sigmoid(g)
            if step<3:
                pdf.append(probability*remaining)
                remaining=remaining*(1.-probability)
            else:
                pdf.append(remaining)
        cdf=torch.stack(pdf,dim=1).cumsum(dim=1)
        mask=cdf>=1.
        choice=mask.float().argmax(dim=1)
        choice=torch.where(mask.any(dim=1),choice,3)
        counts.scatter_add_(0,choice,torch.ones_like(choice))
        if index in checkpoints:
            state.update(hidden=[h[:,-1:,:].detach() for h in hidden],
                         choice=choice.detach(),cdf=cdf.detach(),gate_dtype=str(gates[0].dtype))
    handle=model.model.register_forward_hook(hook)
    inp=torch.tensor([row['prompt_ids']],device='cuda')
    cache=None
    try:
        with torch.inference_mode():
            for index in range(1025):
                result=model(inp,past_key_values=cache,use_cache=True,exit_at_step=3,logits_to_keep=1)
                cache=result.past_key_values
                if index in checkpoints:
                    selected=int(state['choice'][0])
                    hypothetical=model.lm_head(state['hidden'][selected]).float().reshape(-1)
                    fixed=result.logits.float().reshape(-1)
                    p=fixed.double().softmax(-1)
                    q=hypothetical.double().softmax(-1)
                    middle=(p+q)/2
                    js=.5*((p*(p.clamp_min(1e-300).log()-middle.clamp_min(1e-300).log())).sum()+
                            (q*(q.clamp_min(1e-300).log()-middle.clamp_min(1e-300).log())).sum())
                    snapshots.append(dict(step=index,default_selected_step_zero_based=selected,
                        default_selected_recurrent_depth=selected+1,gate_dtype=state['gate_dtype'],
                        cdf=state['cdf'].float().cpu().tolist(),JSD_to_fixed_R4_nats=max(0.,float(js)),
                        default_top1=int(hypothetical.argmax()),fixed_R4_top1=int(fixed.argmax())))
                inp=torch.tensor([[row['teacher_ids'][index]]],device='cuda')
    finally:
        handle.remove()
    write(output/'hf_threshold_audit.json',dict(id=row['id'],calibration_only=True,teacher_steps=1025,
        actual_forward_exit_at_step=3,default_choice_counts_R1_R2_R3_R4=counts.cpu().tolist(),
        snapshots=snapshots,source_sha256=sha(MODEL/'modeling_ouro.py'),
        lock_sha256=sha(ROOT/'hf_threshold_audit_lock.json'),
        benchmark_outputs_changed=False,
        scope='Hypothetical default selector on one teacher trajectory. No alternative benchmark accuracy or author-harness claim.'))
