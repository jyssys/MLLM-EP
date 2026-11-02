"""Recover native AR comparisons only through explicit equality gates."""
import copy
from .common import *
from .configs import finish
from .evidence import compare,status,rows
from .extension_discovery import enqueue
from .extension_runtime_controls import terminal

PATH=ROOT/'state/ar_recovery_20261102.json'


def compiled_controls():
    marker=ROOT/'state/ar_compiled_no_graph_controls_20261102.json'
    if marker.exists():return
    a='ext_qwen_bi_compiled_no_graph_qualification8_s0_v1';b='ext_qwen_bi_compiled_no_graph_qualification8_s1_v1'
    if not terminal([a,b]):return
    gate=compare(a,b) if status(a)==status(b)=='COMPLETE' else dict(passed=False)
    if not gate['passed']:
        write(marker,dict(time=time.time(),status='BLOCKED_N8_REPEAT',gate=gate));return
    source=json.loads((ROOT/'configs'/('ext_qwen_bi_eager_qualification32_reference_v1.json')).read_text())
    dataset=source['dataset'];prompt_ids=source['prompt_ids'];new=[]
    for label,arm,chunk,rate in [('reference','PD_G1',8000,2.),('repeat','PD_G1',8000,4.),
        ('dp2','DP2',8000,4.),('dp2_repeat','DP2',8000,4.),('c4096','PD_G1',4096,4.)]:
        c=copy.deepcopy(source);c.update(name=f'ext_qwen_bi_compiled_no_graph_qualification32_{label}_v1',
            ar_enforce_eager=False,ar_cudagraph_mode='NONE',ar_execution_family='BI_COMPILED_NO_GRAPH',
            arm=arm,chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk),phase='final_correctness',
            gpus=[0,1] if arm=='PD_G1' else [4,5],timeout=1500)
        c=finish(c,32,rate=rate);c.update(dataset=dataset,prompt_ids=prompt_ids,
            data_scope=dict(original_pool_records=32,offered_count=32,explicit_cyclic_reuse=False))
        new.append(c)
    enqueue(new);write(marker,dict(time=time.time(),status='QUALIFICATION32_PENDING',gate8=gate,configs=[c['name'] for c in new],
        scope='One-variable official compilation control with CUDA graphs disabled. Do not pool its cost with Eager or default graphs.'))


def advance():
    import importlib
    from . import ar_compiled_recovery
    importlib.reload(ar_compiled_recovery);ar_compiled_recovery.advance()
    compiled_controls()
    state=json.loads(PATH.read_text()) if PATH.exists() else dict(stage='WAIT_TRUE_BI_EAGER24')
    compiled_path=ROOT/'state/ar_compiled_recovery_20261102.json'
    compiled=json.loads(compiled_path.read_text()) if compiled_path.exists() else {}
    if compiled.get('gates',{}).get('repeat',{}).get('passed') and compiled.get('gates',{}).get('dp2',{}).get('passed') and state['stage'] in ('CAPACITY_AND_CALIBRATION','QPS_PILOTS'):
        from .campaign import terminal as observed
        path=ROOT/'state/skipped_jobs.json';skips=json.loads(path.read_text()) if path.exists() else {}
        for name in state.get('costs',[])+state.get('pilots',[]):
            if observed(name)=='PENDING':skips[name]=dict(time=time.time(),status='NOT_RUN_SUPERSEDED_AR_EXECUTION_FAMILY',
                reason='The compiled graph-NONE control passed both N32 PD repeat and PD/DP equality. Preserve Eager failures and prioritize the qualified comparable control within the user deadline.')
        write(path,skips);state.update(stage='SUPERSEDED_BY_QUALIFIED_COMPILED_CONTROL',time=time.time());write(PATH,state)
        return state
    if state['stage']=='WAIT_TRUE_BI_EAGER24':
        a='ext_qwen_bi_eager_pd_qualification24_s0_v2';b='ext_qwen_bi_eager_pd_qualification24_s1_v2'
        if not terminal([a,b]):return state
        gate=compare(a,b) if status(a)==status(b)=='COMPLETE' else dict(passed=False)
        if not gate['passed']:
            state.update(stage='BLOCKED_TRUE_BI_EAGER24',gate=gate);write(PATH,state);return state
        source=json.loads((ROOT/'configs'/(a+'.json')).read_text());new=[]
        pool=json.loads((ROOT/'data/test_qwen8_L8000.json').read_text())[32:64]
        dataset='extension_qualification32_qwen8_L8000.json';write(ROOT/'data'/dataset,pool)
        for label,arm,chunk,rate in [('reference','PD_G1',8000,2.),('repeat','PD_G1',8000,4.),('dp2','DP2',8000,4.),
            ('c1024','PD_G1',1024,4.),('c4096','PD_G1',4096,4.)]:
            c=copy.deepcopy(source);c.update(name=f'ext_qwen_bi_eager_qualification32_{label}_v1',arm=arm,chunk=chunk,
                chunk_label='full' if chunk==8000 else str(chunk),phase='final_correctness',gpus=[0,1] if arm=='PD_G1' else [4,5],
                seed=2026110507,arrival_seed_base=20261121,timeout=1200)
            c=finish(c,32,rate=rate,start=0)
            c.update(dataset=dataset,prompt_ids=[r['prompt_id'] for r in pool],data_scope=dict(original_pool_records=32,offered_count=32,explicit_cyclic_reuse=False))
            new.append(c)
        for start,rate in enumerate((1.1432986051,2.2865972102)):
            c=copy.deepcopy(source);c.update(name=f'ext_qwen_bi_compiled_no_graph_qualification8_s{start}_v1',
                ar_enforce_eager=False,ar_cudagraph_mode='NONE',ar_execution_family='BI_COMPILED_NO_GRAPH',phase='long_correctness',gpus=[2,3],timeout=900)
            c=finish(c,8,rate=rate);new.append(c)
        enqueue(new);state.update(stage='QUALIFICATION32',gate24=gate,configs=[c['name'] for c in new],source=source);write(PATH,state)
    if state['stage']=='QUALIFICATION32' and terminal(state['configs']):
        reference='ext_qwen_bi_eager_qualification32_reference_v1';gates={}
        for label in ('repeat','dp2','c1024','c4096'):
            name=f'ext_qwen_bi_eager_qualification32_{label}_v1'
            gates[label]=compare(reference,name) if status(reference)==status(name)=='COMPLETE' else dict(passed=False,status=status(name))
        if not gates['repeat']['passed']:
            state.update(stage='BLOCKED_QUALIFICATION32',gates=gates);write(PATH,state);return state
        source=state['source'];new=[];names=[]
        from .search import configured_base
        for chunk in (8000,1024,4096):
            if chunk!=8000 and not gates['c'+str(chunk)]['passed']:continue
            c=copy.deepcopy(source);c.update(name=f'capacity_qwen8_ext_bi_eager_c{chunk}_p_v1',worker='ar_capacity',capacity_mode='P_ONLY',
                phase='independent_capacity',P_slots=8,gpus=[0],chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk))
            c=finish(c,8);c['P_slots']=8;c['jobs']=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4,8)]
            new.append(c);names.append(c['name'])
        c=copy.deepcopy(source);c.update(name='capacity_qwen8_ext_bi_eager_prefix_D_v1',worker='ar_prefix_capacity',phase='independent_capacity',
            capacity_mode='PREPARED_D',gpus=[0],jobs=[dict(batch=b,cap=200,repeats=3) for b in (1,4,8,16,32)])
        c=finish(c,32);new.append(c);names.append(c['name'])
        c=copy.deepcopy(source);c.update(name='ext_calibration_qwen8_bi_eager_l8000_v1',phase='calibration',gpus=[0,1],
            calibration_serial=True,ar_execution_family='BI_EAGER',timeout=1800)
        c=finish(c,16,split='calibration');new.append(c);names.append(c['name'])
        enqueue(new);state.update(stage='CAPACITY_AND_CALIBRATION',gates32=gates,costs=names);write(PATH,state)
    if state['stage']=='CAPACITY_AND_CALIBRATION' and terminal(state['costs']):
        from .analyze import capacity,percentile
        from ..looped_pd_bottleneck_crossover.analyze_serving import token_intervals
        samples=[r for r in capacity() if r.get('execution_family')=='BI_EAGER' and r['model']=='qwen8' and r['length']==8000]
        p=[r for r in samples if r['capacity_mode']=='P_ONLY']
        d=[r for r in samples if r['capacity_mode']=='PREPARED_D' and r['output_cap']==200]
        calibration='ext_calibration_qwen8_bi_eager_l8000_v1';values=rows(calibration)
        if not p or not d or status(calibration)!='COMPLETE' or len(values)!=16:
            state.update(stage='BLOCKED_CAPACITY_OR_CALIBRATION',P_samples=len(p),D_samples=len(d),calibration_requests=len(values));write(PATH,state);return state
        chunks={}
        for chunk in sorted({r['chunk'] for r in p}):
            group=[r for r in p if r['chunk']==chunk];best=max(r['finite_requests_per_s_mean'] for r in group)
            selected=min((r for r in group if r['finite_requests_per_s_mean']>=.95*best),key=lambda r:r['batch'])
            chunks[str(chunk)]=dict(mu_P=selected['finite_requests_per_s_mean'],P_slots=selected['batch'],source=selected)
        bestD=max(d,key=lambda r:r['finite_requests_per_s_mean']);bestC=max(chunks,key=lambda k:chunks[k]['mu_P'])
        nonfull=[k for k in chunks if k!='8000'];tuned=max(nonfull,key=lambda k:chunks[k]['mu_P']) if nonfull else bestC
        cell=dict(model='qwen8',length=8000,chunks=chunks,best_chunk=int(bestC),tuned_chunk=int(tuned),D={'200':dict(mu_D=bestD['finite_requests_per_s_mean'],source=bestD)})
        reference_file='capacity_references_qwen8_bi_eager_extension.json';write(ROOT/'configs'/reference_file,{'qwen8_L8000':cell})
        slo_file='frozen_SLO_qwen8_bi_eager_extension.json';slo_path=ROOT/'configs'/slo_file
        if not slo_path.exists():
            limits=dict(TTFT_ms=2*percentile([r['ttft_ms'] for r in values],.95),TPOT_ms=2*percentile([r['tpot_ms'] for r in values],.95),
                ITL_ms=2*percentile([v for r in values for v in token_intervals(r)[0]],.99),requests=16,calibration=calibration,
                rule='Same preregistered 2x serial request-p95 and token-p99 rule. New execution family, disjoint sources, frozen before its held-out QPS search.',
                frozen_before_heldout_search=time.time(),not_a_hard_guarantee=True)
            write(slo_path,{'qwen8_L8000':limits})
        source=state['source'];mu=min(max(v['mu_P'] for v in chunks.values()),bestD['finite_requests_per_s_mean']);new=[]
        for chunk in sorted(map(int,chunks)):
            for factor in (.25,.5,.75,1.,1.25,1.5,2.):
                c=copy.deepcopy(source);c.update(name=f'ext_pilot_qwen8_bi_eager_c{chunk}_f{str(factor).replace(".","p")}_v1',
                    phase='qps_pilot',arm='PD_G1',chunk=chunk,chunk_label='full' if chunk==8000 else str(chunk),gpus=[0,1],
                    capacity_reference_file=reference_file,SLO_file=slo_file,reference_capacity=mu,rate_factor=factor)
                c=finish(c,24,rate=mu*factor);c['P_slots']=chunks[str(chunk)]['P_slots'];new.append(c)
        reference='ext_qwen_bi_eager_qualification32_reference_v1';dp_gates={}
        for label in ('full_repeat','c1024_b2048','c4096_b8192'):
            name='ext_qwen_bi_eager_dp32_'+label+'_v1'
            gate=compare(reference,name) if status(name)=='COMPLETE' else dict(passed=False,status=status(name))
            dp_gates[label]=gate
            if not gate['passed']:continue
            template=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            for factor in (.75,1.,1.25):
                c=copy.deepcopy(template);c.update(name=f'ext_pilot_qwen8_bi_eager_dp_{label}_f{str(factor).replace(".","p")}_v1',
                    phase='qps_pilot',capacity_reference_file=reference_file,SLO_file=slo_file,reference_capacity=mu,rate_factor=factor,gpus=[0,1])
                c=finish(c,24,rate=mu*factor);new.append(c)
        enqueue(new);state.update(stage='QPS_PILOTS',pilots=[c['name'] for c in new],reference_capacity=mu,
            capacity_reference_file=reference_file,SLO_file=slo_file,DP_gates=dp_gates);write(PATH,state)
    if state['stage']=='QPS_PILOTS' and terminal(state['pilots']):
        state.update(stage='PILOTS_COMPLETE',time=time.time());write(PATH,state)
    return state


if __name__=='__main__':advance()
