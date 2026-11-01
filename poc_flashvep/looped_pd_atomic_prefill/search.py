"""Advance the declared discovery stages from actual completed evidence."""
import statistics
import random
import math
from .common import *
from .configs import finish
from .evidence import records,rows,status,qualify,compare,atomic,transfer,full_work

STATE=ROOT/'state/search_stage.json'
LOCAL_REFERENCE_VERSION='v2'
HUGINN_LONG_VERSION='v3'


def configured_base(model,*args,**kwargs):
    value=base(model,*args,**kwargs)
    if model=='huginn':
        value.update(kv_blocks=24576,token_budget=value['length']*2,
            memory_control='Huginn full-Prompt OOM control: common smaller KV pool and two-Prompt aggregate budget for all methods.')
    if model=='qwen8':value['ar_batch_invariant']=False
    return value


def enqueue(configurations,stage):
    names=[save_config(c) for c in configurations]
    path=ROOT/'configs/study_plan.json';plan=json.loads(path.read_text())
    write(path,dict(configs=list(dict.fromkeys(plan['configs']+names))))
    write(STATE,dict(time=time.time(),stage=stage,generated=names))
    append(ROOT/'state/search_transitions.jsonl',dict(time=time.time(),stage=stage,configs=names))
    print('SEARCH_STAGE',stage,len(names),flush=True)
    if not names:
        write(STATE,dict(time=time.time(),stage='GPU_SEARCH_COMPLETE_ANALYSIS_PENDING',empty_stage=stage,
            reason='No eligible next GPU configuration. This is not a successful regime result.'))
    return bool(names)


def allowed():
    path=ROOT/'state/model_gates.json'
    return json.loads(path.read_text()) if path.exists() else {}


def short_gates():
    gates={};new=[]
    for model in ('qwen8','ouro','huginn'):
        candidate=f'gate_{model}_native_full_l512_pd_g1_'+('v2' if model=='qwen8' else 'v1')
        reference='gate_qwen8_native_full_l512_dp2_v1' if model=='qwen8' else f'gate_{model}_legacy_g1_l512_v1'
        if status(candidate)!='COMPLETE' or status(reference)!='COMPLETE':
            gates[model]=dict(passed=False,stage='SHORT',reason='Missing successful gate namespace',candidate_status=status(candidate),reference_status=status(reference));continue
        result=qualify(model,reference,candidate);gates[model]=result
        if not result['passed']:continue
        length=8000 if model=='qwen8' else 8192 if model=='ouro' else 3900
        count=8;cap=16
        if model=='qwen8':
            for arm in ('PD_G1','DP2'):
                c=base(model,f'qual_{model}_full_l{length}_{arm.lower()}_v1',length,cap,arm=arm,phase='long_correctness')
                new.append(finish(c,count))
        else:
            c=base(model,f'qual_{model}_local_l{length}_v1',length,cap,worker='reference_worker',phase='long_correctness',state_probe=True)
            c['gpus']=c['gpus'][:1];new.append(finish(c,count))
            for label,chunk,wave in (('full',length,False),('legacy2048',2048,True)):
                c=base(model,f'qual_{model}_{label}_l{length}_v1',length,cap,chunk=chunk,chunk_label=label,wavefront=wave,
                    phase='long_correctness',state_probe=True,verify_transfer=True)
                new.append(finish(c,count))
    write(ROOT/'state/model_gates.json',gates)
    return enqueue(new,'LONG_QUALIFICATION')


def native_reference_qualification(model,candidate,reference):
    folder=ROOT/'raw'/reference;refs=records(folder/'references.jsonl')
    actual=rows(candidate);ref=refs[-1] if refs else None
    expected={int(k.rsplit('_q',1)[1]):v for k,v in ref['outputs'].items()} if ref else {}
    exact=bool(actual) and len(actual)==len(expected) and all(r['output_ids']==expected[r['sample_id']]['output_ids'] and
        r['exit_depths']==expected[r['sample_id']]['depths'] for r in actual)
    def state_map(name):
        return {(r['seed'],r['prompt_tokens']):(r['hidden_sha256'],r['planes']) for p in (ROOT/'raw'/name).glob('prompt_states_*.jsonl') for r in records(p)}
    a=state_map(candidate);b=state_map(reference)
    state_exact=bool(actual) and all(a.get((r['seed'],r['input_tokens']))==b.get((r['seed'],r['input_tokens'])) and (r['seed'],r['input_tokens']) in a for r in actual)
    result=dict(model=model,reference=reference,candidate=candidate,output_sequences=len(actual),exact_output_parity=exact,
        exact_valid_KV_hidden_state=state_exact,transfer=transfer(candidate),work=full_work(candidate),atomic=atomic(candidate),
        scope='Internal finite original-runtime reference, not original HF or answer accuracy.')
    result['passed']=exact and state_exact and all(result[k]['passed'] for k in ('transfer','work','atomic'))
    return result


def long_gates():
    gates=allowed()
    for model,gate in list(gates.items()):
        if not gate['passed']:continue
        length=8000 if model=='qwen8' else 8192 if model=='ouro' else 3900
        candidate=f'qual_{model}_full_l{length}_'+('pd_g1_batchinv_v2' if model=='qwen8' else HUGINN_LONG_VERSION if model=='huginn' else 'v1')
        reference=f'qual_{model}_full_l{length}_pd_g1_batchinv_repeat_v3' if model=='qwen8' else f'qual_{model}_local_l{length}_'+(HUGINN_LONG_VERSION if model=='huginn' else LOCAL_REFERENCE_VERSION)
        if any(status(n)!='COMPLETE' for n in (candidate,reference)):
            gates[model]=dict(passed=False,stage='LONG',reason='Long qualification did not complete');continue
        if model=='qwen8':
            result=qualify(model,reference,candidate)
            result['colocated_output_comparison']=compare(f'qual_{model}_full_l{length}_dp2_batchinv_v2',candidate)
            result['cross_mode_lossless_proven']=result['colocated_output_comparison']['passed']
            result['scope']='Native Static-PD repeat qualification. Colocated output fidelity is a separate unresolved limitation. No lossless PD-vs-DP claim is permitted.'
        else:
            result=native_reference_qualification(model,candidate,reference)
            legacy=f'qual_{model}_legacy2048_l{length}_'+(HUGINN_LONG_VERSION if model=='huginn' else 'v1')
            result['legacy_output_comparison']=compare(legacy,candidate) if status(legacy)=='COMPLETE' else dict(passed=False,reason='Legacy long gate failed')
            result['passed'] &= result['legacy_output_comparison']['passed']
        gates[model]=result
    write(ROOT/'state/model_gates.json',gates)
    new=[]
    for model,gate in gates.items():
        if not gate['passed']:continue
        length=8000 if model=='qwen8' else 8192 if model=='ouro' else 3900
        for chunk in (256,1024,2048,4096):
            if chunk>=length:continue
            c=configured_base(model,f'qual_{model}_atomic_chunk{chunk}_l{length}_v1',length,16,chunk=chunk,
                chunk_label=str(chunk),phase='chunk_correctness',ar_batch_invariant=model=='qwen8')
            new.append(finish(c,8))
    return enqueue(new,'CHUNK_QUALIFICATION')


def chunk_gates():
    gates=allowed();qualifications={}
    for model,gate in gates.items():
        if not gate['passed']:continue
        length=8000 if model=='qwen8' else 8192 if model=='ouro' else 3900
        ref=f'qual_{model}_full_l{length}_'+('pd_g1_batchinv_v2' if model=='qwen8' else HUGINN_LONG_VERSION if model=='huginn' else 'v1')
        entries={str(length):dict(passed=True,reference=ref,chunk=length)}
        for chunk in (256,1024,2048,4096):
            if chunk>=length:continue
            name=f'qual_{model}_atomic_chunk{chunk}_l{length}_v1'
            result=compare(ref,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
            if model!='qwen8' and result['passed']:result['work']=full_work(name);result['passed'] &= result['work']['passed']
            entries[str(chunk)]=dict(chunk=chunk,**result)
        qualifications[model]=entries
    write(ROOT/'state/chunk_gates.json',qualifications)
    new=[]
    for arm in ('PD_G1','DP2'):
        c=configured_base('qwen8',f'qual_qwen8_full_l8000_{arm.lower()}_default_recheck_v3',8000,16,arm=arm,phase='long_correctness')
        new.append(finish(c,8))
    return enqueue(new,'AR_DEFAULT_RECHECK')


def ar_default_recheck():
    gates=allowed();candidate='qual_qwen8_full_l8000_pd_g1_default_recheck_v3'
    if status(candidate)=='COMPLETE':
        result=qualify('qwen8','qual_qwen8_full_l8000_pd_g1_v1',candidate)
        result['second_reference']=compare('qual_qwen8_full_l8000_pd_g1_repeat_v1',candidate)
        result['passed'] &= result['second_reference']['passed']
        result['colocated_output_comparison']=compare('qual_qwen8_full_l8000_dp2_default_recheck_v3',candidate)
        result['colocated_repeat_comparison']=compare('qual_qwen8_full_l8000_dp2_v1','qual_qwen8_full_l8000_dp2_default_recheck_v3')
        result.update(ar_batch_invariant=False,cross_mode_lossless_proven=result['colocated_output_comparison']['passed'] and result['colocated_repeat_comparison']['passed'],
            scope='Three independent default native-PD starts must agree. The failed optional batch-invariant family remains failed and is not used for timing. Colocated fidelity is reported separately.')
    else:result=dict(passed=False,status=status(candidate),scope='Default native-PD recheck did not complete.')
    write(ROOT/'state/qwen8_default_recheck.json',result);gates['qwen8']=result;write(ROOT/'state/model_gates.json',gates)
    if not result['passed']:return enqueue(capacity_configs(gates),'CAPACITY')
    new=[]
    for chunk in (256,1024,2048,4096):
        c=configured_base('qwen8',f'qual_qwen8_atomic_chunk{chunk}_l8000_v1',8000,16,chunk=chunk,chunk_label=str(chunk),phase='chunk_correctness')
        new.append(finish(c,8))
    return enqueue(new,'AR_CHUNK_QUALIFICATION')


def ar_chunk_gates():
    gates=allowed();path=ROOT/'state/chunk_gates.json';values=json.loads(path.read_text())
    ref='qual_qwen8_full_l8000_pd_g1_default_recheck_v3';entries={'8000':dict(passed=True,reference=ref,chunk=8000)}
    for chunk in (256,1024,2048,4096):
        name=f'qual_qwen8_atomic_chunk{chunk}_l8000_v1'
        entries[str(chunk)]=dict(chunk=chunk,**(compare(ref,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))))
    values['qwen8']=entries;write(path,values)
    return enqueue(capacity_configs(gates),'CAPACITY')


def capacity_configs(gates):
    new=[]
    for model,gate in gates.items():
        if not gate['passed']:continue
        for length in sorted({n for n,cap in GRID[model] if n!=16384}):
            caps=sorted({cap for n,cap in GRID[model] if n==length},reverse=True)
            primary=(length==8000 if model=='qwen8' else length==8192 if model=='ouro' else length==3900)
            chunks=[256,1024,2048,4096,length] if primary else [length]
            effective=[]
            for chunk in chunks:
                size=min(chunk,length)
                if size not in effective:effective.append(size)
            for chunk in effective:
                gatefile=ROOT/'state/chunk_gates.json'
                if primary and gatefile.exists() and not json.loads(gatefile.read_text()).get(model,{}).get(str(chunk),{}).get('passed'):continue
                label='full' if chunk==length else str(chunk)
                c=configured_base(model,f'capacity_{model}_l{length}_p_chunk{label}_v1',length,max(caps),chunk=chunk,chunk_label=label,
                    worker='ar_capacity' if model=='qwen8' else 'capacity_worker',phase='independent_capacity',capacity_mode='P_ONLY')
                c['gpus']=c['gpus'][:1];c=finish(c,8)
                c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4,8)]
                # Local slots and source PD slots are explicit in the measured geometry.
                if model!='qwen8':c.update(capacity=8,slots=8)
                new.append(c)
            c=configured_base(model,f'capacity_{model}_l{length}_d_v1',length,max(caps),worker='ar_capacity' if model=='qwen8' else 'capacity_worker',
                phase='independent_capacity',capacity_mode='PREPARED_D')
            c=finish(c,8);c['gpus']=c['gpus'][-1:]
            batches=(1,2,4,8,16) if model=='qwen8' else (1,2,4,8)
            c['jobs']=[dict(batch=b,cap=cap,repeats=3) for cap in caps for b in batches]
            if model=='qwen8':
                c['count']=max(batches)
                c=finish(c,c['count'])
            new.append(c)
    return new


def references():
    from .analyze import capacity
    measurements=capacity();selected={}
    for model,gate in allowed().items():
        if not gate['passed']:continue
        optional=ROOT/'state/optional16k_gate.json'
        optional_ready=optional.exists() and json.loads(optional.read_text()).get('passed')
        for length in sorted({n for n,cap in GRID[model] if n!=16384 or model=='ouro' and optional_ready}):
            p=[r for r in measurements if r['model']==model and r['length']==length and r['capacity_mode']=='P_ONLY']
            if not p:continue
            chunks={}
            for chunk in sorted({r['chunk'] for r in p}):
                candidates=[r for r in p if r['chunk']==chunk];best=max(r['finite_requests_per_s_mean'] for r in candidates)
                choice=min((r for r in candidates if r['finite_requests_per_s_mean']>=.95*best),key=lambda r:r['batch'])
                chunks[str(chunk)]=dict(mu_P=choice['finite_requests_per_s_mean'],P_slots=min(choice['batch'],4) if model=='qwen8' else choice['batch'],
                    cohort_batch=choice['batch'],source=choice)
            best_chunk=max(chunks,key=lambda k:chunks[k]['mu_P']);d={}
            nonfull=[k for k in chunks if int(k)!=length]
            tuned_chunk=int(max(nonfull,key=lambda k:chunks[k]['mu_P'])) if nonfull else int(best_chunk)
            for cap in sorted({o for n,o in GRID[model] if n==length}):
                samples=[r for r in measurements if r['model']==model and r['length']==length and r['capacity_mode']=='PREPARED_D' and r['output_cap']==cap]
                if samples:
                    choice=max(samples,key=lambda r:r['finite_requests_per_s_mean']);d[str(cap)]=dict(mu_D=choice['finite_requests_per_s_mean'],source=choice)
            selected[model+'_L'+str(length)]=dict(model=model,length=length,chunks=chunks,best_chunk=int(best_chunk),tuned_chunk=tuned_chunk,D=d,
                scope='Measured finite independent service rates. All-ready D is not an online capacity certificate.')
    write(ROOT/'configs/capacity_references.json',selected);return selected


def serving_config(cell,name,cap,chunk,phase,count,rate=None,start=0,arm='PD_G1'):
    model=cell['model'];length=cell['length'];label='full' if chunk==length else str(chunk)
    c=configured_base(model,name,length,cap,chunk=chunk,chunk_label=label,phase=phase,arm=arm,timeout=5400)
    if phase in ('final_repeat','profile_control'):c['arrival_seed_base']=20261111
    c=finish(c,count,rate=rate,start=start)
    c['P_slots']=cell['chunks'].get(str(chunk),cell['chunks'][str(cell['best_chunk'])])['P_slots']
    c['capacity_reference']=dict(mu_P=cell['chunks'].get(str(chunk),cell['chunks'][str(cell['best_chunk'])])['mu_P'],
        strongest_mu_P=max(r['mu_P'] for r in cell['chunks'].values()),mu_D=cell['D'][str(cap)]['mu_D'])
    return c


def calibration_configs():
    native_mixed_gates()
    selected=references();new=[]
    for key,cell in selected.items():
        if str(cell['length']) not in cell['chunks'] or not cell['D']:continue
        cap=max(map(int,cell['D']));chunk=cell['best_chunk'];name=f'calibration_{key.lower()}_v1'
        c=serving_config(cell,name,cap,chunk,'calibration',16)
        # Use only disjoint calibration prompts. Do not reset the tuned P slots.
        slots=c['P_slots'];c=finish(c,16,split='calibration');c.update(P_slots=slots,calibration_serial=True)
        new.append(c)
    return enqueue(new,'CALIBRATION')


def native_mixed_gates():
    result={}
    for model,length in [('ouro',8192),('huginn',3900)]:
        reference=f'qual_{model}_mixed_reference_l{length}_v1';entries={}
        for chunk in (256,1024):
            name=f'qual_{model}_native_mixed_c{chunk}_l{length}_v1'
            retry=f'qual_{model}_native_mixed_c{chunk}_l{length}_v2'
            if (ROOT/'configs'/(retry+'.json')).exists():name=retry
            if status(reference)!=status(name) or status(name)!='COMPLETE':entries[str(chunk)]=dict(passed=False,status=status(name));continue
            r=compare(reference,name);r['work']=full_work(name)
            calls=sum(v['submissions'] for p in (ROOT/'raw'/name).glob('native_mixed_execution_*.json')
                for v in json.loads(p.read_text())['histogram'] if v['mixed'])
            r.update(actual_mixed_core_calls=calls,scope='Existing native RLT Mixed control. Not a new mechanism.')
            r['passed'] &= r['work']['passed'] and calls>0;entries[str(chunk)]=r
        result[model]=entries
    write(ROOT/'state/native_mixed_gates.json',result);return result


def final_count(rate,start):
    """Use the offered trace itself to ensure a 30-second central window."""
    rng=random.Random(20261111+start);count=1;last=0.
    while count<128 or .7*last<30.:
        last+=rng.expovariate(rate);count+=1
    return count


def freeze_and_pilots():
    from .analyze import percentile
    selected=json.loads((ROOT/'configs/capacity_references.json').read_text());slos={};new=[];matrix=[]
    for key,cell in selected.items():
        name=f'calibration_{key.lower()}_v1';values=rows(name)
        if status(name)!='COMPLETE' or len(values)!=16:continue
        from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
        itls=[v for r in values for v in token_intervals(r)[0]]
        slos[key]=dict(TTFT_ms=2*percentile([r['ttft_ms'] for r in values],.95),TPOT_ms=2*percentile([r['tpot_ms'] for r in values],.95),
            ITL_ms=2*percentile(itls,.99),requests=16,calibration=name,rule='2x independent serial-calibration p95 request latency; 2x token ITL p99.',
            frozen_before_heldout_search=time.time(),not_a_hard_guarantee=True)
        model=cell['model'];length=cell['length'];chunks=list(dict.fromkeys([length,cell['tuned_chunk']]))
        for cap in sorted(map(int,cell['D']),reverse=True):
            mu=min(max(v['mu_P'] for v in cell['chunks'].values()),cell['D'][str(cap)]['mu_D'])
            primary=length==(8000 if model=='qwen8' else 8192 if model=='ouro' else 3900)
            factors=(.25,.5,.75,1.,1.25,1.5,2.) if primary else (.5,1.,1.5)
            for chunk in chunks:
                for index,factor in enumerate(factors):
                    label='full' if chunk==length else str(chunk)
                    name=f'pilot_{model}_l{length}_o{cap}_c{label}_f{str(factor).replace(".","p")}_v1'
                    c=serving_config(cell,name,cap,chunk,'qps_pilot',24,rate=mu*factor)
                    c.update(rate_factor=factor,reference_capacity=mu,rate_scope='Finite independent-service reference. N24 is a smoke screen only.')
                    new.append(c);matrix.append(dict(name=name,model=model,length=length,cap=cap,chunk=chunk,QPS=mu*factor,count=24,status='PLANNED_SMOKE_ONLY'))
    path=ROOT/'configs/frozen_SLO.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        def thresholds(values):return {k:{f:r[f] for f in ('TTFT_ms','TPOT_ms','ITL_ms','calibration','requests')} for k,r in values.items()}
        assert thresholds(frozen)==thresholds(slos),'Do not change a frozen SLO'
        slos=frozen
    else:write(path,slos)
    mixed_path=ROOT/'state/native_mixed_gates.json';mixed=json.loads(mixed_path.read_text()) if mixed_path.exists() else {}
    for model,length,cap in [('ouro',8192,200),('ouro',8192,32),('huginn',3900,32)]:
        key=model+'_L'+str(length);cell=selected.get(key)
        if key not in slos or not cell or str(cap) not in cell['D']:continue
        mu=min(max(v['mu_P'] for v in cell['chunks'].values()),cell['D'][str(cap)]['mu_D'])
        variants=[('DP',c,'rlt_worker') for c in dict.fromkeys([length,cell['tuned_chunk']])]
        variants += [('DP_Mixed',int(chunk),'native_mixed_worker') for chunk,gate in mixed.get(model,{}).items() if gate['passed']]
        for arm,chunk,worker in variants:
            for factor in (.75,1.,1.25):
                name=f'pilot_{model}_l{length}_o{cap}_{arm.lower()}_c{chunk}_f{str(factor).replace(".","p")}_v1'
                c=serving_config(cell,name,cap,chunk,'qps_pilot',24,rate=mu*factor,arm=arm)
                c.update(worker=worker,rate_factor=factor,reference_capacity=mu)
                if arm=='DP_Mixed':c.update(scheduler_mode='mixed',wavefront=True,atomic_prompt=False,native_mixed=True)
                new.append(c);matrix.append(dict(name=name,model=model,length=length,cap=cap,chunk=chunk,QPS=mu*factor,count=24,arm=arm,status='PLANNED_BASELINE_TUNING_SMOKE'))
    write(ROOT/'configs/pilot_matrix.json',matrix)
    # The AR motivation case precedes secondary cells. This order is fixed
    # before held-out Serving outcomes become available.
    priority={'qwen8':(8000,200),'ouro':(8192,32),'huginn':(3900,32)}
    new.sort(key=lambda c:(('qwen8','ouro','huginn').index(c['model']),
        (c['length'],c['cap'])!=priority[c['model']],c['arm']!='PD_G1'))
    return enqueue(new,'QPS_PILOTS')


def finalist_configs():
    from .analyze import serving
    runs=serving();selected=json.loads((ROOT/'configs/capacity_references.json').read_text());new=[];choices={}
    for model in ('qwen8','ouro','huginn'):
        from .live_audit import blocked
        if blocked(model):continue
        pool=[r for r in runs if r['model']==model and r['phase']=='qps_pilot' and r['arm']=='PD_G1' and r['eligible_clean']]
        if not pool:continue
        primary=8000 if model=='qwen8' else 8192 if model=='ouro' else 3900
        primary_pool=[r for r in pool if r['length']==primary and (model!='qwen8' or r['output_cap']==200)]
        if model=='qwen8' and primary_pool:pool=primary_pool
        def score(r):
            cell=selected[model+'_L'+str(r['length'])];muD=cell['D'][str(r['output_cap'])]['mu_D'];muP=max(v['mu_P'] for v in cell['chunks'].values())
            growth=r.get('P_queue_half_growth') or 0.;credit=r.get('D_KV_refusal_sample_fraction')
            credit=credit if isinstance(credit,(float,int)) else 1.
            safe=r['D_TPOT_margin_pass'] and credit<=.01
            pressure=growth>=1. and isinstance(r.get('P_queue_slope_per_s'),(int,float)) and r['P_queue_slope_per_s']>=.05*r['realized_QPS']
            near=isinstance(r.get('P_completion_rate_central'),(int,float)) and r['P_completion_rate_central']>=.85*muP
            return (int(safe and pressure and near),int(safe and pressure),int(safe),muD/muP,min(growth,10.),-abs(r['target_QPS']/muP-1.25))
        chosen=max(pool,key=score);cell=selected[model+'_L'+str(chosen['length'])];cap=chosen['output_cap']
        chosen_config=json.loads((ROOT/'configs'/(chosen['name']+'.json')).read_text())
        transport_limits=chosen_config.get('native_transport_limits',{})
        runtime_controls={k:chosen_config[k] for k in ('native_transport_limits','D_token_budget','D_kv_blocks') if k in chosen_config}
        # A low-rate D-safe screen is not evidence about P saturation. When
        # no P-pressure screen exists, retain a negative diagnostic around
        # the strongest measured P rate rather than repeating only low QPS.
        p_screen=bool(score(chosen)[1])
        center=chosen['target_QPS'] if p_screen else max(v['mu_P'] for v in cell['chunks'].values())
        rates=[center*.8,center,center*1.2]
        chunk_pool=[r for r in runs if r['model']==model and r['length']==cell['length'] and r['output_cap']==cap and
            r['phase']=='qps_pilot' and r['arm']=='PD_G1' and r['chunk']<cell['length'] and
            r['transport_control']==chosen['transport_control'] and r['eligible_clean']]
        finite_tuned=cell['tuned_chunk'];serving_tuned=None
        if chunk_pool:
            nearest=min(abs(r['target_QPS']-center) for r in chunk_pool)
            serving_tuned=max((r for r in chunk_pool if abs(r['target_QPS']-center)==nearest),
                key=lambda r:(r['SLO_goodput'],-r['TTFT_ms_p95']))
            cell=dict(cell,tuned_chunk=serving_tuned['chunk'])
        colocated=[r for r in runs if r['model']==model and r['phase']=='qps_pilot' and r['length']==cell['length'] and r['output_cap']==cap and
            r['arm'] in ('DP','DP2','DP_Mixed') and r['eligible_clean'] and isinstance(r.get('SLO_goodput'),(int,float))]
        dp_choice=None
        if colocated:
            nearest=min(abs(r['target_QPS']-center) for r in colocated)
            best=max((r for r in colocated if abs(r['target_QPS']-center)==nearest),key=lambda r:(r['SLO_goodput'],-r['E2E_ms_p95']))
            dp_choice=dict(arm=best['arm'],chunk=best['chunk'],worker='native_mixed_worker' if best['arm']=='DP_Mixed' else 'rlt_worker',source=best['name'])
        choices[model]=dict(smoke=chosen['name'],length=cell['length'],cap=cap,rates=rates,
            P_pressure_screen_found=p_screen,
            native_transport_limits=transport_limits,
            native_runtime_controls=runtime_controls,
            isolated_P_tuned_chunk=finite_tuned,Serving_tuned_chunk=cell['tuned_chunk'],
            Serving_chunk_selection_source=serving_tuned['name'] if serving_tuned else None,
            colocated_choice=dp_choice,reason='Exploratory selection: measured safe P pressure and near-reference service precede an optimistic independent D/P ratio. Decode SLO and credit remain mandatory. Confirmation uses independent starts.',
            scope='Exploratory condition selection. The independent final starts validate it; the N24 screen is not final evidence.')
        for start in range(3):
            for ri,rate in enumerate(rates):
                count=final_count(rate,start)
                variants=[('PD_G1',v) for v in dict.fromkeys([cell['length'],cell['tuned_chunk']])]
                if dp_choice:variants.append((dp_choice['arm'],dp_choice['chunk']))
                for arm,chunk in variants:
                    label='full' if chunk==cell['length'] else str(chunk)
                    name=f'final_{model}_l{cell["length"]}_o{cap}_{arm.lower()}_c{label}_r{ri}_s{start}_v1'
                    c=serving_config(cell,name,cap,chunk,'final_repeat',count,rate=rate,start=start,arm=arm)
                    if arm=='PD_G1':c.update(runtime_controls)
                    if dp_choice and arm==dp_choice['arm']:
                        c['worker']=dp_choice['worker']
                        if arm=='DP_Mixed':c.update(scheduler_mode='mixed',wavefront=True,atomic_prompt=False,native_mixed=True)
                    c['gpus']=[2*start,2*start+1];c.update(rate_index=ri,selection_smoke=chosen['name'])
                    new.append(c)
        # Actual CUDA activity and overhead use an identical final-start cohort.
        for profile_chunk in dict.fromkeys([cell['length'],cell['tuned_chunk']]):
            label='full' if profile_chunk==cell['length'] else str(profile_chunk)
            for profile_start in range(3 if p_screen else 1):
                name=f'profile_{model}_final_center'+('' if label=='full' else '_c'+label)+('' if profile_start==0 else f'_s{profile_start}')+'_v1'
                rate=rates[1];count=final_count(rate,profile_start)
                c=serving_config(cell,name,cap,profile_chunk,'profile_control',count,rate=rate,start=profile_start)
                c.update(runtime_controls)
                c.update(nsys=True,nsys_graph_trace='graph',gpus=[2*profile_start,2*profile_start+1],
                    profile_clean_reference=f'final_{model}_l{cell["length"]}_o{cap}_pd_g1_c{label}_r1_s{profile_start}_v1');new.append(c)
    cell=selected.get('qwen8_L8000')
    if cell and '200' in cell['D'] and not blocked('qwen8'):
        single=next((r for r in json.loads((ROOT/'capacity_summary.json').read_text()) if r['model']=='qwen8' and r['length']==8000 and
            r['chunk']==8000 and r['capacity_mode']=='P_ONLY' and r['batch']==1),None)
        if single:
            mu=min(single['finite_requests_per_s_mean'],cell['D']['200']['mu_D'])
            for start in range(3):
                for factor in ((.75,1.,1.25,1.5) if start==0 else (1.25,)):
                    rate=mu*factor;count=final_count(rate,start)
                    for arm in ('PD_G1','DP2'):
                        name=f'paper_qwen8_l8000_o200_{arm.lower()}_f{str(factor).replace(".","p")}_s{start}_v1'
                        c=serving_config(cell,name,200,8000,'paper_budget_control',count,rate=rate,start=start,arm=arm)
                        c.update(token_budget=8192,P_slots=1,gpus=[2*start,2*start+1],paper_budget_control=True,
                            budget_scope='8192-token budget. P slot=1 prevents a 192-token partial second Prompt. DP2 keeps native chunked Mixed scheduling.',
                            warning='This geometry control is separate from the tuned multi-request P baseline. It cannot establish tuning-resistant headroom by itself.')
                        new.append(c)
    write(ROOT/'configs/finalist_selection.json',choices)
    write(ROOT/'configs/deferred_final_plan.json',dict(configs=[save_config(c) for c in new]))
    qualification=[]
    for model,choice in choices.items():
        cell=dict(selected[model+'_L'+str(choice['length'])],tuned_chunk=choice['Serving_tuned_chunk'])
        cap=choice['cap'];rate=choice['rates'][1]
        arms=[('full_reference','PD_G1',cell['length']),('full_repeat','PD_G1',cell['length'])]
        if cell['tuned_chunk']!=cell['length']:arms.append(('tuned','PD_G1',cell['tuned_chunk']))
        dp=choice.get('colocated_choice')
        if dp:arms.append(('colocated',dp['arm'],dp['chunk']))
        pool=json.loads((ROOT/'data'/f'test_{model}_L{cell["length"]}.json').read_text())
        source=pool[32:64] if len(pool)>=64 else pool[-32:]
        assert len(source)==32
        dataset=f'qualification32_{model}_L{cell["length"]}.json';write(ROOT/'data'/dataset,source)
        for label,arm,chunk in arms:
            name=f'qualification32_{model}_{label}_v1'
            c=serving_config(cell,name,cap,chunk,'final_correctness',32,rate=rate,start=101,arm=arm)
            if arm=='PD_G1':c.update(choice.get('native_runtime_controls',{}))
            if label=='colocated':
                c['worker']=dp['worker']
                if arm=='DP_Mixed':c.update(scheduler_mode='mixed',wavefront=True,atomic_prompt=False,native_mixed=True)
            c.update(dataset=dataset,prompt_ids=[r['prompt_id'] for r in source],seed=2026110207,
                qualification_scope='Independent arrival and Sampling seeds. Rows 32-63 when available, otherwise the last 32 distinct optional-pool prompts. N32, full depth, and exact output gates remain unchanged. Earlier byte gates remain separate.')
            qualification.append(c)
    return enqueue(qualification,'FINAL_QUALIFICATION')


def final_qualification():
    deferred=json.loads((ROOT/'configs/deferred_final_plan.json').read_text())['configs'];choices=json.loads((ROOT/'configs/finalist_selection.json').read_text())
    verdicts={};new=[];skip_path=ROOT/'state/skipped_jobs.json';skipped=json.loads(skip_path.read_text()) if skip_path.exists() else {}
    for model,choice in choices.items():
        ref=f'qualification32_{model}_full_reference_v1';repeat=f'qualification32_{model}_full_repeat_v1'
        result=compare(ref,repeat) if status(ref)==status(repeat)=='COMPLETE' else dict(passed=False,reference_status=status(ref),repeat_status=status(repeat))
        if result['passed']:
            result['atomic']=atomic(ref);result['passed'] &= result['atomic']['passed']
            if model!='qwen8':result['work']=full_work(ref);result['passed'] &= result['work']['passed']
        result['settings']={}
        for label in ('tuned','colocated'):
            name=f'qualification32_{model}_{label}_v1'
            r=compare(ref,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
            if r['passed'] and model!='qwen8':r['work']=full_work(name);r['passed'] &= r['work']['passed']
            result['settings'][label]=r
        verdicts[model]=result
    for filename in deferred:
        c=json.loads((ROOT/'configs'/filename).read_text());r=verdicts.get(c['model'],{})
        passed=r.get('passed',False)
        if c['phase']=='paper_budget_control':passed=False;reason='NOT_RUN_UNQUALIFIED_PAPER_GEOMETRY'
        elif c['arm'] in ('DP','DP2','DP_Mixed'):passed &= r.get('settings',{}).get('colocated',{}).get('passed',False);reason='NOT_RUN_BLOCKED_COLOCATED_CORRECTNESS'
        elif c['chunk']!=c['length']:passed &= r.get('settings',{}).get('tuned',{}).get('passed',False);reason='NOT_RUN_BLOCKED_TUNED_CORRECTNESS'
        else:reason='NOT_RUN_BLOCKED_FINAL_CORRECTNESS'
        if passed:new.append(c)
        else:skipped[c['name']]=dict(time=time.time(),status=reason,model=c['model'])
    write(ROOT/'state/final_qualification.json',verdicts);write(skip_path,skipped)
    return enqueue(new,'FINAL_REPEATS_AND_PROFILES')


def advance():
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(stage='SHORT_GATES')
    if state['stage']=='SHORT_GATES':return short_gates()
    if state['stage']=='LONG_QUALIFICATION':return long_gates()
    if state['stage']=='CHUNK_QUALIFICATION':return chunk_gates()
    if state['stage']=='AR_DEFAULT_RECHECK':return ar_default_recheck()
    if state['stage']=='AR_CHUNK_QUALIFICATION':return ar_chunk_gates()
    if state['stage']=='CAPACITY':return calibration_configs()
    if state['stage']=='CALIBRATION':return freeze_and_pilots()
    if state['stage']=='QPS_PILOTS':
        from .optional16k import start_capacity
        if start_capacity():return True
        return refinement_or_finalists()
    if state['stage'] in ('OPTIONAL16K_CAPACITY','OPTIONAL16K_CHUNK_CAPACITY'):
        from .optional16k import calibration
        if calibration():return True
        return refinement_or_finalists()
    if state['stage']=='OPTIONAL16K_CALIBRATION':
        from .optional16k import pilots
        if pilots():return True
        return refinement_or_finalists()
    if state['stage']=='OPTIONAL16K_QPS':return refinement_or_finalists()
    if state['stage']=='FINE_QPS_REFINEMENT':return finalist_configs()
    if state['stage']=='FINAL_QUALIFICATION':return final_qualification()
    if state['stage']=='FINAL_REPEATS_AND_PROFILES':
        from .analyze import capacity,serving
        capacity();serving()
        from .profiles import main as profile_analysis
        profile_analysis()
        from .margin import configurations
        diagnostic=configurations()
        if diagnostic:return enqueue(diagnostic,'D_MARGIN_VALIDATION')
        write(ROOT/'state/D_margin_validation.json',{})
        write(STATE,dict(stage='GPU_SEARCH_COMPLETE_ANALYSIS_PENDING',time=time.time()))
        return False
    if state['stage']=='D_MARGIN_VALIDATION':
        from .margin import analyze
        from .classify import main as classify
        analyze();classify()
        write(STATE,dict(stage='GPU_SEARCH_COMPLETE_ANALYSIS_PENDING',time=time.time()))
        return False
    return False


def refinement_or_finalists():
    from .refinement import configurations
    values=configurations()
    return enqueue(values,'FINE_QPS_REFINEMENT') if values else finalist_configs()
