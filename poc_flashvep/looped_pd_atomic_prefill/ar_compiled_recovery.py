"""Gate the official compiled batch-invariant, graph-NONE AR control."""
import copy
from .common import *
from .configs import finish
from .evidence import compare,status,rows
from .extension_discovery import enqueue
from .extension_runtime_controls import terminal

PATH=ROOT/'state/ar_compiled_recovery_20261102.json'
FAMILY='BI_COMPILED_NO_GRAPH'


def serving_P_slots(selected_batch,qualified_slots):
    return max(selected_batch,qualified_slots)


def retire_optional_looped_tail():
    marker=ROOT/'state/optional_looped_tail_retirement_20261102.json'
    if marker.exists():return
    from .campaign import terminal as observed
    from .extension_discovery import queue_priority
    path=ROOT/'state/skipped_jobs.json';skips=json.loads(path.read_text()) if path.exists() else {}
    retired=[]
    for filename in json.loads((ROOT/'configs/study_plan.json').read_text())['configs']:
        cfg=json.loads((ROOT/'configs'/filename).read_text())
        if cfg['model'] not in ('ouro','huginn') or queue_priority(cfg)<4 or observed(cfg['name'])!='PENDING':continue
        skips[cfg['name']]=dict(time=time.time(),status='NOT_RUN_EXTENSION_PRIORITY_TIME_BUDGET',
            reason='Both selected Looped center-rate three-start comparisons and native D diagnostics are complete. Reserve the remaining bounded budget for AR controls and closeout.')
        retired.append(cfg['name'])
    write(path,skips);write(marker,dict(time=time.time(),retired=retired,
        scope='All unstarted optional Looped tail jobs use the same budget rule. Active, complete, failed, and center-priority namespaces remain unchanged.'))


def P4_startup_recovery(state):
    """Keep the failed P1 run. Recheck the already qualified P4 geometry."""
    failure='ext_pilot_qwen8_bi_compiled_none_pd_g1_c4096_f0p25_v1'
    failed_log=ROOT/'raw'/failure/'prefill.log'
    if state['stage']=='P4_STARTUP_RECHECK_V3':retire_optional_looped_tail()
    if state['stage']=='PILOTS_COMPLETE' and not state.get('P4_recovery') and status(failure)=='FAILED' and failed_log.exists():
        if 'an illegal memory access was encountered' not in failed_log.read_text():return state
        cfg=json.loads((ROOT/'configs'/(failure+'.json')).read_text())
        if cfg['P_slots']!=1 or rows(failure):return state
        snapshot=ROOT/'state/ar_compiled_before_P4_startup_recovery_20261102.json'
        if not snapshot.exists():write(snapshot,state)
        source=state['source'];assert source['P_slots']==4
        new=[]
        for label,chunk,factor in (('full',8000,.7),('c4096',4096,1.2)):
            c=copy.deepcopy(source);c.update(name=f'ext_qwen_bi_compiled_P4_recheck32_{label}_v2',
                phase='final_correctness',chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk),
                gpus=[0,1] if chunk==8000 else [2,3],dataset_pool=source['dataset'],nsys=False,nsys_P_only=False,timeout=900)
            c=finish(c,32,rate=state['reference_capacity']*factor);c['P_slots']=4;new.append(c)
        enqueue(new)
        state.update(stage='P4_STARTUP_RECHECK',P4_recovery=dict(failure=failure,checks=[c['name'] for c in new],
            scope='P1 engine startup failed before any request. P4 matches the earlier N32-qualified engine capacity. The exact kernel fault remains unlocalized.'))
        confirmation=ROOT/'state/extension_confirmation_20261102.json'
        if confirmation.exists():
            value=json.loads(confirmation.read_text())
            if value.get('qwen8',{}).get('stage')=='NOT_RUN_NO_ELIGIBLE_PILOT':
                write(ROOT/'state/ar_confirmation_before_P4_recovery_20261102.json',value.pop('qwen8'))
                write(confirmation,value)
        write(PATH,state)
    if state['stage']=='BLOCKED_P4_STARTUP_RECHECK' and not state['P4_recovery'].get('corrected_dataset_checks'):
        reference='ext_qwen_bi_compiled_no_graph_qualification32_reference_v1'
        expected={r['sample_id']:r for r in rows(reference)}
        checked=[rows(name) for name in state['P4_recovery']['checks']]
        incomparable=bool(checked) and all(len(v)==32 and all(r['input_ids']!=expected[r['sample_id']]['input_ids'] for r in v) for v in checked)
        if incomparable:
            write(ROOT/'state/ar_P4_invalid_dataset_comparisons_20261102.json',dict(time=time.time(),gates=state['P4_recovery']['gates'],
                matched_input_rows=0,reason='finish() selected the default test pool instead of the earlier qualification subset. These are incomparable inputs, not measured numerical failures.',
                new_test_pool_Full_vs_chunk=compare(*state['P4_recovery']['checks'])))
            new=[];source=state['source']
            for i,(label,chunk,arm,factor) in enumerate((('full',8000,'PD_G1',.7),('c4096',4096,'PD_G1',1.2),('dp2',8000,'DP2',1.),('dp2_repeat',8000,'DP2',1.3))):
                c=copy.deepcopy(source);c.update(name=f'ext_qwen_bi_compiled_P4_recheck32_{label}_v3',
                    phase='final_correctness',arm=arm,chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk),
                    gpus=[2*i,2*i+1],dataset_pool=source['dataset'],nsys=False,nsys_P_only=False,timeout=900)
                c=finish(c,32,rate=state['reference_capacity']*factor);c['P_slots']=4 if arm=='PD_G1' else 32;new.append(c)
            enqueue(new);state['P4_recovery']['corrected_dataset_checks']=[c['name'] for c in new]
            state['stage']='P4_STARTUP_RECHECK_V3';write(PATH,state)
    if state['stage'] in ('P4_STARTUP_RECHECK','P4_STARTUP_RECHECK_V3') and terminal(state['P4_recovery'].get('corrected_dataset_checks',state['P4_recovery']['checks'])):
        reference='ext_qwen_bi_compiled_no_graph_qualification32_reference_v1'
        gates={name:compare(reference,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
            for name in state['P4_recovery'].get('corrected_dataset_checks',state['P4_recovery']['checks'])}
        state['P4_recovery']['gates']=gates
        if all(v['passed'] for v in gates.values()):
            state['source']['ar_execution_family']='BI_COMPILED_NO_GRAPH_P4'
            state.update(stage='CAPACITY_AND_CALIBRATION',pilot_suffix='p4_v2')
        else:state['stage']='BLOCKED_P4_STARTUP_RECHECK'
        write(PATH,state)
    return state


def advance():
    state=json.loads(PATH.read_text()) if PATH.exists() else dict(stage='QUALIFICATION32')
    state=P4_startup_recovery(state)
    reference='ext_qwen_bi_compiled_no_graph_qualification32_reference_v1'
    if state['stage']=='QUALIFICATION32':
        names=[f'ext_qwen_bi_compiled_no_graph_qualification32_{label}_v1' for label in ('reference','repeat','dp2','dp2_repeat','c4096')]
        if not terminal(names):return state
        gates={}
        for label in ('repeat','dp2','dp2_repeat','c4096'):
            name=f'ext_qwen_bi_compiled_no_graph_qualification32_{label}_v1'
            gates[label]=compare(reference,name) if status(reference)==status(name)=='COMPLETE' else dict(passed=False,status=status(name))
        if not gates['repeat']['passed']:
            state.update(stage='BLOCKED_QUALIFICATION32',gates=gates);write(PATH,state);return state
        source=json.loads((ROOT/'configs'/(reference+'.json')).read_text());new=[]
        for chunk in (8000,4096):
            if chunk!=8000 and not gates['c4096']['passed']:continue
            c=copy.deepcopy(source);c.update(name=f'capacity_qwen8_ext_bi_compiled_none_c{chunk}_p_v1',worker='ar_capacity',
                capacity_mode='P_ONLY',phase='independent_capacity',gpus=[0],chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk))
            c=finish(c,8);c['P_slots']=8;c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4,8)];new.append(c)
        c=copy.deepcopy(source);c.update(name='capacity_qwen8_ext_bi_compiled_none_prefix_D_v1',worker='ar_prefix_capacity',
            phase='independent_capacity',capacity_mode='PREPARED_D',gpus=[0],jobs=[dict(batch=b,cap=200,repeats=3) for b in (1,4,8,16,32)])
        c=finish(c,32);new.append(c)
        c=copy.deepcopy(source);c.update(name='ext_calibration_qwen8_bi_compiled_none_l8000_v1',phase='calibration',
            gpus=[0,1],calibration_serial=True,timeout=1800)
        c=finish(c,16,split='calibration');new.append(c)
        enqueue(new);state.update(stage='CAPACITY_AND_CALIBRATION',gates=gates,source=source,costs=[c['name'] for c in new]);write(PATH,state)
    if state['stage']=='CAPACITY_AND_CALIBRATION' and not state.get('profile_tool_controls'):
        source=state['source'];new=[];clean='ext_profile_tool_qwen_bi_compiled_none_P_clean_v1'
        for profiled in (False,True):
            c=copy.deepcopy(source);c.update(name='ext_profile_tool_qwen_bi_compiled_none_P_only_v1' if profiled else clean,
                phase='profile_control' if profiled else 'profile_tool_control',gpus=[0,1],nsys=False,
                nsys_P_only=profiled,profile_clean_reference=clean if profiled else None,timeout=1200)
            c=finish(c,4,rate=2.);new.append(c)
        enqueue(new);state['profile_tool_controls']=[c['name'] for c in new];state['costs']+=state['profile_tool_controls'];write(PATH,state)
    if state['stage']=='CAPACITY_AND_CALIBRATION' and terminal(state['costs']):
        from .analyze import capacity,percentile
        from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
        samples=[r for r in capacity() if r.get('execution_family')==FAMILY and r['model']=='qwen8' and r['length']==8000]
        p=[r for r in samples if r['capacity_mode']=='P_ONLY'];d=[r for r in samples if r['capacity_mode']=='PREPARED_D' and r['output_cap']==200]
        calibration='ext_calibration_qwen8_bi_compiled_none_l8000_v1';values=rows(calibration)
        if not p or not d or status(calibration)!='COMPLETE' or len(values)!=16:
            state.update(stage='BLOCKED_CAPACITY_OR_CALIBRATION',P_samples=len(p),D_samples=len(d),calibration_requests=len(values));write(PATH,state);return state
        chunks={}
        for chunk in sorted({r['chunk'] for r in p}):
            group=[r for r in p if r['chunk']==chunk];best=max(r['finite_requests_per_s_mean'] for r in group)
            selected=min((r for r in group if r['finite_requests_per_s_mean']>=.95*best),key=lambda r:r['batch'])
            chunks[str(chunk)]=dict(mu_P=selected['finite_requests_per_s_mean'],P_slots=selected['batch'],source=selected)
        bestD=max(d,key=lambda r:r['finite_requests_per_s_mean']);bestC=max(chunks,key=lambda k:chunks[k]['mu_P'])
        tuned=4096 if '4096' in chunks else int(bestC)
        cell=dict(model='qwen8',length=8000,chunks=chunks,best_chunk=int(bestC),tuned_chunk=tuned,
            D={'200':dict(mu_D=bestD['finite_requests_per_s_mean'],source=bestD)})
        reference_file='capacity_references_qwen8_bi_compiled_none_extension.json';write(ROOT/'configs'/reference_file,{'qwen8_L8000':cell})
        slo_file='frozen_SLO_qwen8_bi_compiled_none_extension.json'
        if not (ROOT/'configs'/slo_file).exists():
            write(ROOT/'configs'/slo_file,{'qwen8_L8000':dict(TTFT_ms=2*percentile([r['ttft_ms'] for r in values],.95),
                TPOT_ms=2*percentile([r['tpot_ms'] for r in values],.95),ITL_ms=2*percentile([v for r in values for v in token_intervals(r)[0]],.99),
                requests=16,calibration=calibration,rule='Same preregistered 2x serial request-p95 and token-p99 rule. Separate qualified execution family.',
                frozen_before_heldout_search=time.time(),not_a_hard_guarantee=True)})
        mu=min(max(v['mu_P'] for v in chunks.values()),bestD['finite_requests_per_s_mean']);new=[];source=state['source']
        dp=state['gates']['dp2']['passed'] and state['gates']['dp2_repeat']['passed']
        for arm in (('PD_G1','DP2') if dp else ('PD_G1',)):
            for chunk in sorted(map(int,chunks)):
                # C4096 DP is not yet a direct qualified control. Do not infer
                # its equality from either Full DP or chunked pure PD.
                if arm=='DP2' and chunk!=8000:continue
                for factor in (.25,.5,.75,1.,1.25,1.5,2.):
                    c=copy.deepcopy(source);c.update(name=f'ext_pilot_qwen8_bi_compiled_none_{arm.lower()}_c{chunk}_f{str(factor).replace(".","p")}_{state.get("pilot_suffix","v1")}',
                        phase='qps_pilot',arm=arm,chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk),gpus=[0,1],
                        capacity_reference_file=reference_file,SLO_file=slo_file,reference_capacity=mu,rate_factor=factor)
                    c=finish(c,24,rate=mu*factor)
                    c['P_slots']=serving_P_slots(chunks[str(chunk)]['P_slots'],source['P_slots']) if arm=='PD_G1' else 32
                    new.append(c)
        enqueue(new);state.update(stage='QPS_PILOTS',pilots=[c['name'] for c in new],reference_capacity=mu,
            capacity_reference_file=reference_file,SLO_file=slo_file);write(PATH,state)
    if state['stage']=='QPS_PILOTS' and state.get('P4_recovery') and not state.get('screening_pilots'):
        planned=[json.loads((ROOT/'configs'/(name+'.json')).read_text()) for name in state['pilots']]
        state['screening_pilots']=[c['name'] for c in planned if c['rate_factor'] in (.75,1.,1.25)]
        state['pilot_completion_scope']='Nine prospectively selected coarse screens, three factors per method. Other planned factors remain a lower-priority budget tail.'
        enqueue(planned);write(PATH,state)
    if state['stage']=='QPS_PILOTS' and terminal(state.get('screening_pilots',state['pilots'])):
        state.update(stage='SCREENING_PILOTS_COMPLETE' if state.get('screening_pilots') else 'PILOTS_COMPLETE',time=time.time());write(PATH,state)
    return state


if __name__=='__main__':advance()
