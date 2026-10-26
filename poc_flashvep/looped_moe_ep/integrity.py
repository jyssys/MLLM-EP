"""Check saved distributed evidence without launching any GPU work."""
import json
from .common import ROOT,MODELS,write
from .analyze import completed,rows

def main():
    checks=[]
    def check(name,value,detail):
        checks.append(dict(check=name,pass_check=bool(value),detail=detail))
        assert value,(name,detail)
    runs=completed()
    check('All mandatory jobs complete',len(runs)==42,len(runs))
    rank_count=0
    for run in runs:
        config=json.loads((run/'request.rank0.json').read_text())['config']
        world=len(config['gpus']);model=config['model']
        ownership=[]
        for rank in range(world):
            f=json.loads((run/f'fidelity.rank{rank}.json').read_text())
            check(f'{run.name}/rank{rank}/native_fidelity',f['pass_fidelity'] and f['token_exact'],
                dict(relative_l2=f['relative_l2'],delta_nll=f['delta_nll']))
            check(f'{run.name}/rank{rank}/physical_mapping',f['physical_gpu']==config['gpus'][rank],f['physical_gpu'])
            expected=18 if model=='foil1' else 112
            check(f'{run.name}/rank{rank}/full_recurrence',sum(f['observed_site_calls'].values())==expected,
                f['observed_site_calls'])
            check(f'{run.name}/rank{rank}/real_nccl',bool(f['nccl_version']) and f['ep_degree']==world,f['nccl_version'])
            ownership.append(json.loads((run/f'ownership.rank{rank}.json').read_text()))
            rank_count+=1
        for index,site in enumerate(ownership[0]):
            all_ids=[e for owner in ownership for e in owner[index]['owned']]
            check(f'{run.name}/{site["site"]}/unique_owner_union',
                sorted(all_ids)==list(range(site['logical_experts'])),all_ids)
            check(f'{run.name}/{site["site"]}/native_tensor_subset',
                all(t['exact_native_subset'] for owner in ownership for t in owner[index]['tensors']),
                'Exact native BF16 parameter rows')
    for model in MODELS:
        for world in [4,8]:
            run=ROOT/'raw'/f'{model}_ep{world}_shape_quality_v2'
            q=rows(run/'engineering_quality.rank0.jsonl')
            check(f'{model}/EP{world}/engineering_105',len(q)==105 and all(r['pass'] for r in q),len(q))
            ready=ROOT/'raw'/f'{model}_ep{world}_ready_probe_v1'
            data=json.loads((ready/'ready_probe.rank0.json').read_text())['cases']
            check(f'{model}/EP{world}/independent_ready_output',len(data)==6 and all(c['token_exact'] for c in data),
                'B1/B2/B4 x synchronized/staggered, 8 generated tokens')
            steady=ROOT/'raw'/f'{model}_ep{world}_steady_generation_v1'
            data=rows(steady/'steady_generation.rank0.jsonl')
            full=[r for r in data if r['cap']==128]
            check(f'{model}/EP{world}/warm_output_identity',len(full)==4 and all(r['ids']==full[0]['ids'] for r in full),
                'Four 128-token uninstrumented generations')
    write(ROOT/'quality/integrity.json',dict(status='PASS',completed_jobs=len(runs),distributed_rank_runs=rank_count,
        checks=checks,scope='Saved evidence checks. Not downstream task accuracy or optimized-method E2E.'))
    print('INTEGRITY',len(checks),'checks',rank_count,'rank runs')

if __name__=='__main__':main()
