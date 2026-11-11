"""Summarize finite valid-state cohorts without claiming online capacity."""
import json
from .common import *
from .analyze import summary


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    path=folder/'cohorts.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    groups=[]
    for batch in (1,2,4):
        local=[r for r in rows if r['job']['batch']==batch and r['status']=='COMPLETE']
        rates=[batch/r['prefill_host_service_s'] for r in local]
        decode=[batch/r['decode_native_boundary_service_s'] for r in local if r['decode_native_boundary_service_s']>0]
        groups.append(dict(batch=batch,completed_repeats=len(local),P_requests_per_s=summary(rates),
            prepared_D_requests_per_s=summary(decode),
            P_service_seconds=summary(r['prefill_host_service_s'] for r in local),
            prepared_D_service_seconds=summary(r['decode_native_boundary_service_s'] for r in local),
            parity_and_work=bool(local) and all(r['native_output_parity'] and r['full_depth_rows']==r['full_depth_rows_observed'] for r in local),
            blocked_rows=[r for r in rows if r['job']['batch']==batch and r['status']!='COMPLETE']))
    valid=[g for g in groups if g['completed_repeats']>=3 and g['parity_and_work']]
    best=max(valid,key=lambda g:g['P_requests_per_s']['p50']) if valid else None
    value=dict(name=name,model=cfg['model'],length=cfg['length'],cap=cfg['cap'],groups=groups,
        strongest_finite_P_reference=best,
        scope='Host elapsed service with final CUDA synchronization. P excludes first Coda. Prepared D includes first Coda and Decode. Three cohorts share one process. This is not a kernel union or sustainable capacity.')
    write(folder/'capacity_analysis.json',value);return value


def main():
    initialize();plan=json.loads((ROOT/'state/capacity_discovery_plan.json').read_text())
    values=[analyze(n) for n in plan['jobs']]
    write(ROOT/'analysis/independent_capacity_references.json',values)
    print(json.dumps([dict(name=v['name'],best=v['strongest_finite_P_reference']) for v in values]))


if __name__=='__main__':main()
