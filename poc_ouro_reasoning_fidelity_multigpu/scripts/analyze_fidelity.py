"""Freeze limits on calibration inputs, then inspect untouched confirmation differences."""
import argparse
import csv
import sys
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def metrics(reference,actual):
    a,b=reference.astype(np.float64).reshape(-1),actual.astype(np.float64).reshape(-1)
    return dict(max_abs=float(np.max(np.abs(a-b))),relative_L2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-20)))

def logit_metrics(reference,actual):
    a,b=reference.astype(np.float64).reshape(-1),actual.astype(np.float64).reshape(-1)
    p=np.exp(a-a.max());p/=p.sum()
    q=np.exp(b-b.max());q/=q.sum()
    middle=(p+q)/2
    jsd=.5*np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(middle,1e-300)))+.5*np.sum(q*np.log(np.maximum(q,1e-300)/np.maximum(middle,1e-300)))
    # Descending stable order preserves the lowest-ID argmax tie rule.
    ia,ib=np.argsort(-a,kind='stable')[:5],np.argsort(-b,kind='stable')[:5]
    return metrics(a,b)|dict(JSD_nats=max(0.,float(jsd)),reference_top5=ia.tolist(),actual_top5=ib.tolist(),
        centered_logit_relative_L2=float(np.linalg.norm((a-a.mean())-(b-b.mean()))/max(np.linalg.norm(a-a.mean()),1e-20)),
        reference_margin=float(a[ia[0]]-a[ia[1]]),actual_margin=float(b[ib[0]]-b[ib[1]]),
        top1_equal=int(ia[0])==int(ib[0]),top5_overlap=len(set(ia)&set(ib))/5)

def compare(row,tp,step):
    reference_folder='hf457_additional_v1' if row.get('dataset')=='math100' else 'hf457_fidelity_v1'
    reference=np.load(ROOT/f'results/{reference_folder}/teacher_{file_id(row["id"])}.step{step}.npz')
    result=[]
    for rank in range(tp):
        native=np.load(ROOT/f'results/native_fidelity_TP{tp}_v1/teacher_{file_id(row["id"])}.step{step}.rank{rank}.npz')
        first=rank*(16//tp)
        expected_KV=reference['KV'][:,:,first:first+2,:]
        l=logit_metrics(reference['logits'],native['logits'])
        h=metrics(reference['hidden'],native['hidden'])
        k=metrics(expected_KV,native['KV'])
        k['per_slot_relative_L2']=[metrics(expected_KV[i],native['KV'][i])['relative_L2'] for i in range(192)]
        k['worst_slot_relative_L2']=max(k['per_slot_relative_L2'])
        k['worst_slot']=int(np.argmax(k['per_slot_relative_L2']))
        result.append(dict(id=row['id'],split=row['split'],TP=tp,rank=rank,step=step,
            logits=l,hidden=h,KV=k,scope='SDPA versus native FA3, BF16, exact teacher prefix and logical KV mapping.'))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibrate-only',action='store_true')
    args=parser.parse_args()
    inputs=read(ROOT/'results/fidelity_inputs.json')+read(ROOT/'results/fidelity_additional_inputs.json')
    rule=read(ROOT/'fidelity_extension_lock.json')['tolerance_rule']
    freeze=ROOT/'results/fidelity_tolerance_lock.json'
    if not freeze.exists():
        records=[]
        for row in inputs:
            if row['split']!='calibration':continue
            for tp in [1,2,4]:
                for step in [0,1,4,32,128,1024]:
                    records.extend(compare(row,tp,step))
        mapping={'logits_relative_L2':('logits','relative_L2'),'hidden_relative_L2':('hidden','relative_L2'),
            'KV_relative_L2':('KV','relative_L2'),'next_token_JSD':('logits','JSD_nats')}
        limits={}
        for name,(group,metric) in mapping.items():
            maximum=max(r[group][metric] for r in records)
            threshold=max(rule[name]['floor'],rule['calibration_multiplier']*maximum)
            limits[name]=dict(calibration_max=maximum,threshold=threshold,ceiling=rule[name]['ceiling'],
                calibration_pass=threshold<=rule[name]['ceiling'])
        write(freeze,dict(limits=limits,confirmation_read=False,
            reference_version='4.57.1',author_version_failed=True,inputs_sha256=sha(ROOT/'results/fidelity_inputs.json')))
        write(ROOT/'results/fidelity_calibration.json',records)
    if args.calibrate_only:
        frozen=read(freeze)
        if not all(r['calibration_pass'] for r in frozen['limits'].values()):
            print('Calibration planning ceiling failed. Confirmation differences remain unread.',flush=True)
            raise SystemExit(2)
        return
    frozen=read(freeze)
    records=[]
    for row in inputs:
        for tp in [1,2,4]:
            for step in [0,1,4,32,128,1024]:
                records.extend(compare(row,tp,step))
    limits=frozen['limits']
    for r in records:
        r['PASS']=r['logits']['relative_L2']<=limits['logits_relative_L2']['threshold'] and r['hidden']['relative_L2']<=limits['hidden_relative_L2']['threshold'] and r['KV']['relative_L2']<=limits['KV_relative_L2']['threshold'] and r['logits']['JSD_nats']<=limits['next_token_JSD']['threshold']
    natural=[]
    HF_natural={}
    for path in (ROOT/'results/hf457_fidelity_v1').rglob('*cap*.json'):
        row=read(path)
        if isinstance(row,dict) and {'id','cap','ids'}<=set(row):
            HF_natural[(row['id'],row['cap'])]=row
    for tp in [1,2,4]:
        folder=ROOT/f'results/native_fidelity_TP{tp}_v1'
        for path in sorted(folder.glob('natural_*_repeat0.json')):
            a=read(path)
            if (a['id'],a['cap']) not in HF_natural:continue
            b=HF_natural[(a['id'],a['cap'])]
            n=min(len(a['ids']),len(b['ids']))
            index=next((i for i in range(n) if a['ids'][i]!=b['ids'][i]),None)
            repeat=folder/path.name.replace('repeat0','repeat1')
            natural.append(dict(id=a['id'],TP=tp,cap=a['cap'],first_divergence=index,
                native_tokens=len(a['ids']),HF_tokens=len(b['ids']),
                native_finish=a['finish_reason'],HF_finish=b['finish_reason'],
                same_config_repeat_equal=read(repeat)['ids']==a['ids'] if repeat.exists() else None))
    write(ROOT/'results/fidelity_comparisons.json',records)
    write(ROOT/'results/natural_fidelity_comparisons.json',natural)
    repeats=[]
    for tp in [1,2,4]:
        for step in [0,1,4,32,128,1024]:
            for rank in range(tp):
                folder=ROOT/f'results/native_fidelity_TP{tp}_v1'
                a=np.load(folder/f'teacher_60.step{step}.rank{rank}.npz')
                b=np.load(folder/f'teacher_repeat_60.step{step}.rank{rank}.npz')
                repeats.append(dict(TP=tp,step=step,rank=rank,
                    fields={name:metrics(a[name],b[name])|dict(bitwise_equal=bool(np.array_equal(a[name],b[name]))) for name in ['logits','hidden','KV']}))
    write(ROOT/'results/native_same_config_numerical_repeats.json',repeats)
    summary=dict(calibration_pass=all(r['calibration_pass'] for r in limits.values()),
        confirmation_pass=all(r['PASS'] for r in records if r['split']=='confirmation'),
        functional_confirmation_pass=all(r['logits']['JSD_nats']<=limits['next_token_JSD']['threshold'] for r in records if r['split']=='confirmation'),
        confirmation_evaluated=True,original_limit_sha256=sha(freeze),
        same_config_short_repeats_pass=all(r['same_config_repeat_equal'] for r in natural if r['same_config_repeat_equal'] is not None),
        same_config_numerical_bitwise=all(v['bitwise_equal'] for row in repeats for v in row['fields'].values()),
        natural_comparisons=natural,reference_version='4.57.1',author_reference_version_status='HF_REFERENCE_BLOCKED_4.54.1',
        note='Natural BF16 branch changes require margin-based diagnosis. This summary alone cannot establish a consequential model bug.')
    summary['strict_state_gate_pass']=summary['calibration_pass'] and summary['confirmation_pass']
    summary['reference_runtime_verdict']='REFERENCE_RUNTIME_PARTIAL' if summary['functional_confirmation_pass'] and summary['same_config_short_repeats_pass'] and summary['same_config_numerical_bitwise'] else 'NUMERICAL_MISMATCH_UNRESOLVED'
    write(ROOT/'results/fidelity_summary.json',summary)
    print(summary)

if __name__=='__main__':
    main()
