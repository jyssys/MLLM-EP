"""Check recorded normal completion and credit restoration in held-out runs."""
import json
from .common import ROOT,write
from .analysis import records


def main():
    target=ROOT/'analysis/heldout_normal_lifetime_credit_audit.json'
    if target.exists():
        original=json.loads(target.read_text())
        if original.get('schema_version')!=2:
            preserved=ROOT/'analysis/heldout_normal_lifetime_credit_audit_v1_scope_failure.json'
            assert not preserved.exists()
            write(preserved,original)
    rows=[]
    for cfgpath in sorted((ROOT/'configs').glob('heldout_*.json')):
        cfg=json.loads(cfgpath.read_text())
        if not (cfg.get('phase','').startswith('FROZEN_HELDOUT') or
                cfg.get('phase')=='POSTHOC_MATCHED_AGGREGATE_AND_GEOMETRY_CONTROL'):
            continue
        folder=ROOT/'raw'/cfg['name'];receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        if cfg['arm']!='PD_G1':
            rows.append(dict(name=cfg['name'],evidence='NOT_MEASURED',
                scope='Aggregated DP2 has no campaign P/D reservation manager. Normal worker exit and output qualification are separate. No PD credit state is inferred.'))
            continue
        checks={};values={};not_applicable=[]
        for role in ('prefill','decode'):
            for kind in ('KV_lifetime','reservation_lifetime'):
                if kind=='reservation_lifetime' and not cfg.get('dynamic_layer_extension'):
                    not_applicable.append(kind+'_'+role+' has no prospective layer manager')
                    continue
                path=folder/f'{kind}_{role}.json'
                key=kind+'_'+role
                value=json.loads(path.read_text()) if path.exists() else None
                values[key]=value
                checks[key]=value is not None and value.get('passed') is True
            events=[e for path in folder.glob(f'extension_{role}_*.jsonl') for e in records(path)]
            pools=[e for e in events if e.get('event')=='HIDDEN_GPU_POOL_CLOSE']
            values[role+'_pool_close']=pools
            if cfg.get('dynamic_layer_extension'):
                checks[role+'_hidden_pool_close']=bool(pools) and all(e.get('all_ACKed') and not e['pending_requests'] for e in pools)
            else:
                not_applicable.append(role+'_hidden_pool_close has no hidden transport pool')
        rows.append(dict(name=cfg['name'],checks=checks,passed=all(checks.values()),
            observations=values,not_applicable=not_applicable,evidence='TRACE_ANALYSIS',
            scope='Recorded before worker shutdown, after output drain and hidden ACK cleanup. This validates normal completion only. It does not certify cancellation, GPU failure recovery, exclusive memory traffic or long-run sustainable capacity.'))
    measured=[r for r in rows if 'passed' in r]
    write(target,dict(schema_version=2,rows=rows,
        qualified_PD_runs=len(measured),passed_PD_runs=sum(r['passed'] for r in measured),
        evidence='TRACE_ANALYSIS',all_PD_normal_lifetimes_pass=all(r['passed'] for r in measured) if measured else None))
    assert all(r['passed'] for r in measured),[r['name'] for r in measured if not r['passed']]
    print('HELDOUT_NORMAL_LIFETIME_CREDIT_PASS',len(measured),flush=True)


if __name__=='__main__':
    main()
