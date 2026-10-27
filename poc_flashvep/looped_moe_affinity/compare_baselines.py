"""Compare matched hardware rollouts. Keep process-start uncertainty explicit."""
import json
import numpy as np
from .common import ROOT
from .aggregate_quality import csv_write
from .aggregate_timing import ci
from .native_resume import directory,provenance

def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def main():
    native=[];comparisons=[];verification=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        base=directory(model)
        for b in [1,2,4,8,16]:
            for length in [128,512,2048]:
                key=dict(model=model,batch=b,prefix_tokens=length)
                generation=records(base/f'native_generation_b{b}_p{length}.rank0.jsonl')
                forward=base/f'native_forward_b{b}_p{length}.rank0.json'
                if len(generation)!=3 or not forward.exists():
                    native.append(dict(key,status='NOT_COMPLETE',evidence='NOT_AVAILABLE'));continue
                row=dict(key,status='PASS',evidence='LIVE_NATIVE_ONE_GPU',output_cap=32,eos_enabled=False,
                    decode_policy='Full-prefix recomputation, not cached decode',native_attempt=base.name)
                row.update(provenance(base,b,length))
                for name,values in [('e2e_ms',[v['e2e_ms'] for v in generation]),
                        ('ttft_ms',[v['ttft_ms'] for v in generation]),('tpot_ms',[v['tpot_ms'] for v in generation]),
                        ('prefill_ms',json.loads(forward.read_text())['clean_ms'])]:
                    row.update({name+'_'+k:v for k,v in ci(values).items()})
                row['output_tokens_s']=b*32*1000/row['e2e_ms_mean'];native.append(row)
                for ep in [4,8]:
                    epbase=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'
                    for policy,stem in [('static','generation'),('source_owner','generic_generation')]:
                        values=records(epbase/f'{stem}_b{b}_p{length}.rank0.jsonl')
                        if len(values)!=3:continue
                        mean=float(np.mean([v['e2e_ms'] for v in values]))
                        comparisons.append(dict(key,ep=ep,policy=policy,native_e2e_ms=row['e2e_ms_mean'],ep_e2e_ms=mean,
                            native_to_ep_e2e_speedup=row['e2e_ms_mean']/mean,
                            native_source_attempt=row['native_source_attempt'],native_allocator_conf=row['native_allocator_conf'],
                            allocator_matched=row['native_allocator_conf']=='DEFAULT',
                            token_exact_native=all(v['generated_ids']==generation[0]['generated_ids'] for v in values),
                            evidence='LIVE_MATCHED_SHAPE_FREE_RUNNING_GREEDY_CAP32',
                            qualification='One GPU versus multiple GPUs. Same global batch. Not an equal-GPU-budget DP comparison. '+
                                ('Allocator settings differ for this recovered native cell.' if row['native_allocator_conf']!='DEFAULT' else 'Allocator settings match.')))
        for ep in [4,8]:
            starts=[]
            for start in [1,2]:
                path=ROOT/'raw'/f'{model}_ep{ep}_quiet_verify_start{start}_v1/paired_verify.rank0.jsonl'
                values=records(path)
                if len(values)!=12:continue
                static=[v['e2e_ms'] for v in values if v['policy']=='static']
                generic=[v['e2e_ms'] for v in values if v['policy']=='source_owner']
                ratio=float(np.mean(static)/np.mean(generic));starts.append(ratio)
                verification.append(dict(model=model,ep=ep,start=start,static_e2e_ms=float(np.mean(static)),
                    generic_e2e_ms=float(np.mean(generic)),static_to_generic_speedup=ratio,
                    rollouts_per_policy=6,order='|'.join(values[0]['order']),physical_gpu_group='4-7' if ep==4 and start==1 else f'0-{ep-1}',
                    evidence='LIVE_COUNTERBALANCED_INDEPENDENT_PROCESS',
                    qualification='Only two independent starts. This does not establish a precise restart confidence interval.'))
    csv_write(ROOT/'native_baseline.csv',native)
    csv_write(ROOT/'native_ep_comparison.csv',comparisons)
    csv_write(ROOT/'counterbalanced_generic.csv',verification)

if __name__=='__main__':main()
