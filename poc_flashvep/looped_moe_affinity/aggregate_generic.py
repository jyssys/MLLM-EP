"""Report actual generic metadata controls without calling them loop-aware."""
import json
import numpy as np
from .common import ROOT
from .aggregate_quality import csv_write
from .aggregate_timing import ci

def main():
    output=[];best=[];rng=np.random.default_rng(20261027)
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [4,8]:
            base=ROOT/'raw'/f'{model}_ep{ep}_generic_metadata_controls_v1'
            for b in [1,2,4,8,16]:
                for length in [128,512,2048]:
                    path=base/f'generic_b{b}_p{length}.rank0.jsonl'
                    parity=base/f'output_parity_b{b}_p{length}.rank0.json'
                    if not path.exists() or not parity.exists():continue
                    values=[json.loads(line) for line in path.read_text().splitlines()]
                    same=json.loads(parity.read_text())['policy_exact_to_static']
                    by_policy={p:[v for v in values if v['policy']==p] for p in ['static','replicated_counts','replicated_counts_source_owner']}
                    if any(len(v)!=3 for v in by_policy.values()):continue
                    static=float(np.mean([v['e2e_ms'] for v in by_policy['static']]))
                    rows=[]
                    for policy,items in by_policy.items():
                        row=dict(model=model,ep=ep,batch=b,prefix_tokens=length,policy=policy,
                            token_exact_to_static=same[policy],count_allgathers_per_moe=1 if policy=='static' else 0,
                            actual_dispatch_combine=True,downstream_quality='NOT_SEPARATELY_BENCHMARKED_GENERIC_CONTROL',
                            evidence='LIVE_TRUE_EP_GENERIC_CONTROL_NOT_LOOP_AWARE')
                        for name,data in [('e2e_ms',[v['e2e_ms'] for v in items]),('ttft_ms',[v['ttft_ms'] for v in items]),
                                ('tpot_ms',[v['tpot_ms'] for v in items]),('prefill_ms',items[0]['prefill_ms'])]:
                            row.update({name+'_'+k:v for k,v in ci(data).items()})
                        row['static_to_control_e2e_speedup']=static/row['e2e_ms_mean']
                        control=np.asarray([v['e2e_ms'] for v in items])
                        original=np.asarray([v['e2e_ms'] for v in by_policy['static']])
                        bootstrap=(rng.choice(original,(4000,3),replace=True).mean(1)/
                            rng.choice(control,(4000,3),replace=True).mean(1))
                        if policy=='static':bootstrap=np.ones(4000)
                        row['speedup_ci95_low']=float(np.quantile(bootstrap,.025))
                        row['speedup_ci95_high']=float(np.quantile(bootstrap,.975))
                        row['speedup_ci_scope']='Independent resampling of serial warm repetitions. Not paired process-restart uncertainty.'
                        output.append(row);rows.append(row)
                    eligible=[r for r in rows if r['token_exact_to_static']]
                    selected=min(eligible,key=lambda r:r['e2e_ms_mean'])
                    best.append(dict(selected,selection='Minimum observed mean among exact-output generic controls in this process.',
                        limitation='Winner selection is not held-out certification. This is not a production DeepEP comparison.'))
    csv_write(ROOT/'generic_metadata_controls.csv',output)
    csv_write(ROOT/'strongest_measured_generic.csv',best)

if __name__=='__main__':main()
