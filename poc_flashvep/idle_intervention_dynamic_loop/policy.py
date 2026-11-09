"""Use frozen measured costs. Keep admission and execution models separate."""
from .pilot import *
from .analyze import percentile
from collections import defaultdict


def total_bound(name):
    data=summarize(name)
    assert data is not None,'A cost cell did not complete: '+name
    values=list(data['per_request_sums']['decode'].values())
    value=percentile(values,.99)
    return value+max(1.,.2*value)


def candidate(model,width,loops,token=False):
    names=[f'p_{model}_tile{width}_loops1_serial8']
    if loops==2:names=[f'p2_{model}_tile{width}_loops2_quantum2_serial8']
    if token and width in (32,128,256):
        names.append(f'pt_{model}_suffix512_tile{width}_quantum{1 if width==32 else 2}_serial8')
    groups=[]
    for name in names:
        data=summarize(name)
        if data is None:continue
        groups.extend(g for g in data['groups'] if g['role']=='decode'
            and g['tokens']==width and (g['loops']==loops or g['loops']==0))
    if not groups:return None
    if not any(g['loops']==loops for g in groups):return None
    cost=max(g['p99_plus_guard_ms'] for g in groups)
    full=guarded(model,4096 if model=='ouro' else 2048)
    return dict(tokens=width,loops=loops,cost_ms=cost,
        saved_ms=full*width/(4096 if model=='ouro' else 2048)*loops,
        saved_scope='Structural scaling for candidate ranking. Not measured P critical-path savings.',
        provenance=names,cost_scope='Maximum guarded measured cost across relevant terminal, nonterminal, and available Prelude groups.')


def phase_candidates(model,token=False):
    groups=defaultdict(list);names=defaultdict(set);T=4096 if model=='ouro' else 2048
    files=list((ROOT/'raw').glob(f'p*_{model}_*/primitive_cost_summary.json'))
    for path in files:
        value=json.loads(path.read_text())
        for row in value['groups']:
            if row['role']!='decode' or row['tokens'] not in (32,64,128,256,512,1024,2048,3*T//4,T):continue
            if not 0<=row['loops']<=(4 if model=='ouro' else 32):continue
            if row['operation']=='prelude' and not token:continue
            if row['operation'] not in ('prelude','recurrent'):continue
            key=(row['tokens'],row['loops'],row['terminal'],row['operation'])
            groups[key].append(row['p99_plus_guard_ms']);names[key].add(value['name'])
    full=guarded(model,T);result=[]
    for key,values in groups.items():
        width,loops,terminal,operation=key
        result.append(dict(tokens=width,loops=loops,terminal=terminal,operation=operation,
            cost_ms=max(values),saved_ms=full*width/T*max(1,loops),
            provenance=sorted(names[key]),saved_scope='Structural ranking only. Not measured P time.'))
    return result


def configured(model,name,arm,count=24,seed=2026110801,load='heavy'):
    method={'native':'native','zero':'zero','rejection':'dynamic_loop',
        'loop':'dynamic_loop','tile':'dynamic_tile','token':'dynamic_token',
        'static_loop':'static_intervention','static_token':'static_token'}[arm]
    cfg=common_pilot(model,name,method,count,seed)
    timing=configuration(model,name,count,load,seed,method=method)
    for key in ('arrival_offsets','arrival_rate','arrival_seed','load'):cfg[key]=timing[key]
    T,R=cfg['length'],cfg['depth']
    native=metrics(f'cap_{model}_native_calibration128')
    assert native['complete']
    source=ROOT/'raw'/f'cap_{model}_zero_calibration128';batch_samples=defaultdict(list)
    begin=json.loads((source/'measurement_begin.json').read_text())['time']
    for file in source.glob('intervention_*.json'):
        for row in json.loads(file.read_text()).get('records',[]):
            if row['time']>=begin and row['event']=='DECODE_INTERVAL_SAMPLE':
                batch_samples[row['batch']].append(row['interval_ms'])
    table={str(b):percentile(values,.95)+1. for b,values in batch_samples.items() if len(values)>=8}
    cfg.update(native_ITL_target_ms=min(native['frozen_SLO']['ITL_ms'],1.05*native['ITL_p99_ms']),
        feedback_ITL_limit_ms=min(native['frozen_SLO']['ITL_ms'],1.05*native['ITL_p99_ms']),
        decode_reserve_ms=native['ITL_p50_ms'],initial_decode_reserve_by_batch=table,
        minimum_decode_reserve_ms=min(table.values()),
        native_calibration=dict(Original_Native=native['name'],matched_zero=source.name,
            minimum_batch_samples=8,scope='Separate selection arrival seed. Conditional reserves are empirical p95 plus 1 ms, not worst-case bounds.'))
    cfg.update(admission_window_multiplier=8.,empty_admission_window_multiplier=1.,
        multiple_quanta_per_observed_deadline=True,
        defer_token_arming=True,initial_token_budget_selection=True,
        intervention_warmup_widths=sorted({32,64,128,256,512,1024,T//2,3*T//4,T}),intervention_warmup_batches=[1,2],
        D_candidate_widths=sorted({32,64,128,256,512,1024,2048,3*T//4,T}),D_loop_quantum=1,
        policy_revision='calibrated-v1',phase='CALIBRATED_POLICY_SELECTION_NOT_CONFIRMATION')
    # These serial total costs do not predict waiting between Decode-priority yields.
    measured={1:guarded(model,T)}
    for n in ((2,3) if model=='ouro' else (2,4,8)):
        measured[n]=total_bound(f'pf_{model}_loops{n}_quantum1_serial8')
    cfg['remaining_cost_ms']={str(n):measured.get(n,guarded(model,T)*n) for n in range(1,R)}
    cfg['completion_prediction_scope']='Guarded serial D enclosures. Waiting between quanta, future Decode work, and online overlap require direct validation. No oracle speedup claim.'
    if arm in ('loop','static_loop'):
        cfg['phase_aware_costs']=True
        cfg['execution_candidates']=[c for c in phase_candidates(model) if c['tokens']==T]
        cfg['minimum_quantum_ms']=min(c['cost_ms']+cfg['guard_ms'] for c in cfg['execution_candidates'])
    if arm=='rejection':cfg['force_destination_reject']=True
    if arm in ('tile','token','static_token'):
        cfg['phase_aware_costs']=True
        cfg['execution_candidates']=phase_candidates(model,token=arm in ('token','static_token'))
        cfg['minimum_quantum_ms']=min(c['cost_ms']+cfg['guard_ms'] for c in cfg['execution_candidates'])
    if arm in ('token','static_token'):
        cfg['native_frontier']=[0,T-512]
        points=sorted({T//4,T//2,3*T//4,T-512,T-256,T-128,T-64,T-32})
        cfg['boundary_checkpoints']=points
        measured_token=defaultdict(list)
        for file in (ROOT/'raw').glob(f'pt*_{model}_*/primitive_cost_summary.json'):
            data=json.loads(file.read_text());cell=json.loads((ROOT/'configs'/(data['name']+'.json')).read_text())
            frontier=cell.get('native_frontier')
            if frontier and frontier[0]==0:
                measured_token[T-frontier[1]].append(total_bound(data['name']))
        best={suffix:min(values) for suffix,values in measured_token.items()}
        def cost(suffix):
            if suffix in best:return best[suffix]
            nearest=min(best,key=lambda n:abs(n-suffix))
            return best[nearest]*suffix/nearest
        cfg['remaining_cost_ms']={str(T-k):cost(T-k) for k in points}
        cfg['completion_prediction_scope']='Best measured serial total per calibrated suffix length. Other lengths use linear scaling from the closest calibration. Attention context is not linear. Waiting and safe quantum packing remain unvalidated admission estimates.'
        cfg['intervention_warmup_widths']=sorted({T-k for k in points})
        cfg['intervention_warmup_mixed_native']=True
        if arm=='static_token':
            cfg['native_frontier']=[0,T-256]
            cfg['D_tile_tokens']=128;cfg['D_loop_quantum']=2
    if arm=='static_loop':cfg['native_partial_cut']=R-1
    if arm=='native':
        cfg.update(native_path=False,feature_native_wire_v5=False,atomic_prompt=True)
    return cfg


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for arm in ('native','zero','rejection','loop','tile','token'):
            cfg=configured(model,f'v1_{model}_{arm}_pilot24',arm)
            save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';write(p,json.loads(p.read_text())+jobs)


if __name__=='__main__':main()
