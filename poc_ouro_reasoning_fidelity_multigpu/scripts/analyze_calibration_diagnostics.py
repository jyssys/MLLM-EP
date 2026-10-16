"""Describe calibration ablations without changing any frozen state tolerance."""
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_fidelity import metrics,logit_metrics

def main():
    output=ROOT/'results/calibration_diagnostic_summary_v2.json'
    assert not output.exists()
    modes=[]
    for mode in ['native','hf_operator']:
        for step in [0,128,1024]:
            a=np.load(ROOT/f'results/hf457_fidelity_v1/teacher_60.step{step}.npz')
            b=np.load(ROOT/f'results/norm_operator_probe_v1/{mode}.step{step}.rank0.npz')
            modes.append(dict(mode=mode,step=step,hidden=metrics(a['hidden'],b['hidden']),
                logits=logit_metrics(a['logits'],b['logits']),KV=metrics(a['KV'][:,:,:2,:],b['KV'])))
    first=[]
    for row in read(ROOT/'results/calibration_divergent_prefixes.json'):
        hf=np.load(ROOT/f'results/hf_calibration_divergence_v2/first_divergence_{row["id"]}.npz')
        native=np.load(ROOT/f'results/native_calibration_divergence_v1/first_divergence_{row["id"]}.step{row["step"]}.rank0.npz')
        values=logit_metrics(hf['logits'],native['logits'])
        first.append(dict(id=row['id'],step=row['step'],logits=values,
            HF_reproduces_prior_top1=values['reference_top5'][0]==row['reference_next'],
            native_reproduces_prior_top1=values['actual_top5'][0]==row['prior_native_next']))
    weights=read(ROOT/'results/norm_operator_probe_v1/weight_identity.json')[0]
    write(output,dict(calibration_only=True,norm_ablation=modes,first_natural_divergence=first,
        loaded_weight_identity=dict(all_equal=weights['all_equal'],parameters=len(weights['records'])),
        original_state_ceiling_repro=read(ROOT/'results/calibration_ceiling_repro_v1.json'),
        tolerance_changed=False,confirmation_differences_used=False,
        limitation='Norm changes both arithmetic and its kernel path. This ablation does not isolate every rounding or backend difference.'))
    print(first)

if __name__=='__main__':
    main()
