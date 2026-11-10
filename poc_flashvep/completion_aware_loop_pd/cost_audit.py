"""Separate per-operation event sums from complete continuation enclosures."""
import csv
from collections import defaultdict
from .common import *
from .analyze import percentile


def main():
    configure()
    from ..idle_intervention_dynamic_loop import costs
    costs.ROOT=ROOT;costs.configure=configure
    from ..looped_pd_atomic_prefill import evidence
    evidence.ROOT=ROOT
    rows=[];groups=[];verification=[]
    jobs=json.loads((ROOT/'state/refreshed_cost_jobs.json').read_text())
    for name in jobs:
        folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            verification.append(dict(name=name,status='NOT_COMPLETED'));continue
        summary=costs.summarize(name);groups.extend(dict(name=name,model=cfg['model'],**g) for g in summary['groups'])
        expected=set(range(cfg['seed'],cfg['seed']+cfg['count']))
        expected_outputs={r['seed']:r for r in map(json.loads,(OLD/'raw'/cfg['calibration_reference']/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')}
        actual=[r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]
        first_outputs={r['seed']:r['first_token'] for r in actual}
        different=[r['seed'] for r in actual if r['seed'] not in expected_outputs or any(r[k]!=expected_outputs[r['seed']][k] for k in ('input_ids','output_ids','exit_depths'))]
        work=evidence.full_work(name)
        verification.append(dict(name=name,status='COMPLETE',full_work=work,output_divergent_seeds=different,
            output_exact=not different,scope='Exact output comparison with the identified frozen serial calibration. Not a fresh full-state hash.'))
        for p in folder.glob('sparse_device_*.json'):
            v=json.loads(p.read_text());starts={};ends={};last_prompt={}
            for r in v['rows']:
                for request in r['requests']:
                    seed=request['seed']
                    if seed not in expected:continue
                    if r['prompt']:
                        starts[seed]=min(starts.get(seed,float('inf')),r['begin']['time'])
                        last_prompt[seed]=max(last_prompt.get(seed,0),r['end']['time'])
                    elif r['operation']=='coda' and request['start']==cfg['length']-1:
                        ends[seed]=r['end']['time']
            for seed,start in starts.items():
                sums=summary.get('per_request_sums',{}).get(v['role'],{})
                total=sums.get(seed,sums.get(str(seed)))
                rows.append(dict(name=name,model=cfg['model'],role=v['role'],sampling_seed=seed,
                    token_tile=cfg.get('D_tile_tokens'),offloaded_loops=cfg['depth']-cfg['native_partial_cut'],
                    operation_enclosure_sum_ms=total,
                    first_to_last_prompt_GPU_clock_enclosure_ms=1000*(last_prompt[seed]-start),
                    first_prompt_to_first_coda_GPU_clock_enclosure_ms=1000*(ends[seed]-start) if seed in ends else None,
                    first_prompt_GPU_clock_to_first_output_host_ms=1000*(first_outputs[seed]-start),
                    scope='GPU-clock enclosures include intervening host gaps and device waits. They are not kernel unions or safe online execution guarantees.'))
    for filename,values in (('refreshed_primitive_groups',groups),('refreshed_continuation_enclosures',rows)):
        write(ROOT/(filename+'.json'),values)
        if values:
            fields=list(dict.fromkeys(k for r in values for k in r))
            with (ROOT/(filename+'.csv')).open('w') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
    write(ROOT/'refreshed_cost_qualification.json',verification)
    print(json.dumps(dict(complete=len(groups),requests=len(rows),output_failures=sum(bool(v.get('output_divergent_seeds')) for v in verification))))


if __name__=='__main__':main()
