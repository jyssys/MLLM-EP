"""Check measured work and exact state. Dynamic diversity is a separate gate."""
from .common import *


def measured_D_prompt_rows(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    expected=set(range(cfg['seed'],cfg['seed']+cfg['count']));rows={}
    for path in folder.glob('atomic_spans_*.json'):
        data=json.loads(path.read_text())
        if data['role']!='decode':continue
        for row in data['prompt_spans']:
            if row['seed'] in expected and row['time']>=begin:
                amount=row['count']*row['loop_count']
                if amount:rows[row['seed']]=rows.get(row['seed'],0)+amount
    return rows


def qualify(name,reference=None):
    configure()
    from ..looped_pd_atomic_prefill import evidence
    from ..fluidtoken_loop_split.state_comparison import compare
    from ..dynamic_loop_split_serving.lifetime_gate import check
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
        return dict(name=name,complete=False,passed=False)
    before=json.loads((folder/'before_measured.json').read_text())
    after=json.loads((folder/'after_measured.json').read_text())
    key=next(key for key in after if '-decode-' in key)
    actual=after[key]['issued_loop_rows']-before[key]['issued_loop_rows']-cfg['count']*(cfg['cap']-1)*cfg['depth']
    events=[r for path in folder.glob('intervention_*.json') for r in json.loads(path.read_text()).get('records',[])]
    begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    events=[r for r in events if r['time']>=begin]
    expected_seeds=set(range(cfg['seed'],cfg['seed']+cfg['count']))
    commits={r['seed']:r for r in events if r['event']=='SOURCE_COMMIT' and r['seed'] in expected_seeds}
    dynamic=cfg.get('intervention_method') in ('dynamic_loop','dynamic_tile','dynamic_token')
    token=cfg.get('intervention_method')=='dynamic_token'
    expected=(sum((cfg['length']-r['depth'])*cfg['depth'] if token else
        (cfg['depth']-r['depth'])*cfg['length'] for r in commits.values()) if dynamic
        else cfg['count']*(cfg['depth']-cfg.get('native_partial_cut',cfg['depth']))*cfg['length'])
    if not dynamic and cfg.get('native_frontier'):
        c,k=cfg['native_frontier'];expected=cfg['count']*(cfg['depth']-c)*(cfg['length']-k)
    if not dynamic and cfg.get('native_request_cuts'):
        expected=sum((cfg['depth']-cut)*cfg['length'] for cut in cfg['native_request_cuts'])
    result=dict(name=name,complete=True,work=evidence.full_work(name),lifetime=check(name),
        offload=dict(actual_D_prompt_rows=actual,expected_D_prompt_rows=expected,passed=actual==expected,
            requests=len(measured_D_prompt_rows(name)),per_request_D_prompt_rows=measured_D_prompt_rows(name),
            accepted_loop_counts=sorted({cfg['depth']-r['depth'] for r in commits.values()}) if not token else [],
            accepted_suffix_tokens=sorted({cfg['length']-r['depth'] for r in commits.values()}) if token else [],
            adaptive_diversity=dynamic and len({r['depth'] for r in commits.values()})>=2
                and len(commits)<cfg['count']))
    result['actual_dynamic_continuation_tested']=dynamic and bool(commits)
    if cfg.get('verify_transfer'):
        if cfg.get('feature_native_wire_v5'):
            rows=[json.loads(line) for path in folder.glob('KV_checks_*.jsonl') for line in path.read_text().splitlines()]
            selected={r['seed']:r for r in rows if r['seed'] in expected_seeds}
            result['handoff']=dict(compared=len(selected),expected=cfg['count'],
                passed=set(selected)==expected_seeds and all(r['passed'] for r in selected.values()))
        else:result['handoff']=evidence.transfer(name)
    if cfg.get('async_verify_handoff'):
        records=[json.loads(line) for path in folder.glob('async_wire_states_*.jsonl') for line in path.read_text().splitlines()]
        source={r['seed']:r for r in records if r['role']=='prefill' and r['seed'] in expected_seeds}
        destination={r['seed']:r for r in records if r['role']=='decode' and r['seed'] in expected_seeds}
        keys=('KV_sha256','KV_valid_bytes','hidden_sha256')
        different=[seed for seed in expected_seeds&source.keys()&destination.keys()
            if any(source[seed][key]!=destination[seed][key] for key in keys)]
        result['async_handoff']=dict(compared=len(source.keys()&destination.keys()),expected=cfg['count'],
            divergent_seeds=sorted(different),passed=set(source)==set(destination)==expected_seeds and not different,
            scope='Same full valid export KV byte order and hidden bytes. Asynchronous diagnostic, not clean serving.')
    if reference:
        result['output']=evidence.compare(reference,name)
        if cfg.get('state_probe'):result['state']=compare(reference,name)
    result['passed']=all(v.get('passed',v.get('exact',True)) for v in result.values() if isinstance(v,dict))
    write(folder/'qualification.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--reference');a=p.parse_args()
    print(json.dumps(qualify(a.name,a.reference),indent=2))
