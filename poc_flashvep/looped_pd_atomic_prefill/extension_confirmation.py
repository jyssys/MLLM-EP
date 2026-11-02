"""Confirm selected extension screens without changing the bottleneck gates."""
import copy
import random
from .common import *
from .configs import finish
from .evidence import status,compare,full_work,atomic
from .extension_discovery import enqueue
from .extension_runtime_controls import terminal
from .analyze import serving_one

PATH=ROOT/'state/extension_confirmation_20261102.json'


def final_count(rate,start,seed_base=20261131):
    count=128
    while True:
        rng=random.Random(seed_base+start)
        duration=sum(rng.expovariate(rate) for _ in range(count-1))
        if .7*duration>=30:return count
        count+=16


def measured_screen(cfg):
    r,_,_=serving_one(cfg)
    if not r or not r['eligible_clean']:return None
    reference=json.loads((ROOT/'configs'/cfg['capacity_reference_file']).read_text())[cfg['model']+'_L'+str(cfg['length'])]
    muP=max(v['mu_P'] for v in reference['chunks'].values())
    muD=reference['D'][str(cfg['cap'])]['mu_D']
    rate=r['realized_QPS'];p=r.get('P_completion_rate_central') or 0
    clear=all(isinstance(r.get(k),(int,float)) and r[k]<=.01 for k in ('D_KV_refusal_sample_fraction','D_slot_refusal_sample_fraction'))
    growth=(r.get('P_queue_half_growth') or 0)>=1 and (r.get('P_queue_slope_per_s') or 0)>=.05*(rate or 0)
    near=p>=.85*muP;reserve=muD>=1.2*p and (r.get('D_ready_queue_slope_per_s') or 0)<=.05*(rate or 0)
    checks=dict(D_credit_clear=clear,D_TPOT_margin_pass=r['D_TPOT_margin_pass'],P_backlog_growth=growth,
        P_near_finite_service=near,D_independent_finite_reserve=reserve)
    # This score selects a diagnostic, not a positive verdict. CUDA activity,
    # independent starts, and the native staggered-D test remain mandatory.
    score=(all(checks.values()),sum(checks.values()),clear and r['D_TPOT_margin_pass'],
        min(p/max(muP,1e-12),1.),min(max(r.get('P_queue_half_growth') or 0,0),10),p)
    return dict(run=r,checks=checks,selection_score=score,finite_P=muP,finite_D=muD)


def qualify(model,choice):
    source=json.loads((ROOT/'configs'/(choice['run']['name']+'.json')).read_text())
    source.update(seed=2026110607,arrival_seed_base=20261171,extension=True)
    pool=json.loads((ROOT/'data'/f'test_{model}_L{source["length"]}.json').read_text())
    subset=pool[64:96] if len(pool)>=96 else pool[8:40]
    assert len(subset)==32
    dataset=f'extension_confirmation32_{model}_L{source["length"]}.json';write(ROOT/'data'/dataset,subset)
    reference=json.loads((ROOT/'configs'/source['capacity_reference_file']).read_text())[model+'_L'+str(source['length'])]
    chunks=[int(k) for k in reference['chunks'] if int(k)<source['length']]
    tuned=max(chunks,key=lambda k:reference['chunks'][str(k)]['mu_P']) if chunks else source['length']
    output=[]
    variants=[('reference',source['length'],.7),('repeat',source['length'],1.2),('tuned',tuned,1.)]
    if model=='qwen8':variants.append(('dp2',source['length'],1.))
    for index,(label,chunk,factor) in enumerate(variants):
        cfg=copy.deepcopy(source);cfg.update(name=f'ext_confirm32_{model}_{label}_v1',phase='final_correctness',
            chunk=chunk,chunk_label='full' if chunk==source['length'] else str(chunk),
            gpus=[2*index,2*index+1] if model=='qwen8' else [0,1] if model=='ouro' else [4,5],
            verify_transfer=False,state_probe=False,nsys=False,timeout=1800)
        cfg=finish(cfg,32,rate=source['arrival_rate']*factor)
        cfg['P_slots']=source['P_slots']
        if label=='dp2':cfg.update(arm='DP2',P_slots=32)
        cfg.update(dataset=dataset,prompt_ids=[r['prompt_id'] for r in subset],
            data_scope=dict(original_pool_records=32,offered_count=32,explicit_cyclic_reuse=False),
            qualification_scope='Independent prompt subset, arrival seed, and Sampling seed. Existing native valid-KV and hidden-byte gates remain separate.')
        output.append(cfg)
    enqueue(output)
    return dict(stage='QUALIFICATION32',choice=choice,source=source,tuned_chunk=tuned,qualification=[c['name'] for c in output])


def final_jobs(model,item):
    ref,rep,tuned=item['qualification'][:3]
    gate=compare(ref,rep) if status(ref)==status(rep)=='COMPLETE' else dict(passed=False)
    chunk_gate=compare(ref,tuned) if status(ref)==status(tuned)=='COMPLETE' else dict(passed=False)
    if gate['passed']:
        gate['work']=full_work(ref) if model!='qwen8' else dict(passed=True,scope='Native AR, no recurrent work counter.')
        gate['repeat_work']=full_work(rep) if model!='qwen8' else dict(passed=True,scope='Native AR, no recurrent work counter.')
        gate['atomic']=atomic(ref)
        gate['repeat_atomic']=atomic(rep)
        gate['passed'] &= gate['work']['passed'] and gate['repeat_work']['passed'] and gate['atomic']['passed'] and gate['repeat_atomic']['passed']
    if chunk_gate['passed']:
        chunk_gate['work']=full_work(tuned) if model!='qwen8' else dict(passed=True,scope='Native AR.')
        chunk_gate['passed'] &= chunk_gate['work']['passed']
    item.update(repeat_gate=gate,tuned_gate=chunk_gate)
    if not gate['passed']:item['stage']='BLOCKED_QUALIFICATION32';return []
    source=item['source'];center=source['arrival_rate'];new=[];clean_names=[]
    chunks=list(dict.fromkeys([source['length']]+([item['tuned_chunk']] if chunk_gate['passed'] else [])))
    factors=(.8,1.,1.2) if all(item['choice']['checks'].values()) else (1.,)
    item['confirmation_rate_factors']=factors
    item['budget_scope']='Confirm three rates after a positive screen. Otherwise repeat the selected negative diagnostic at one rate with three independent starts.'
    dp_gate=compare(ref,item['qualification'][3]) if model=='qwen8' and status(item['qualification'][3])=='COMPLETE' else dict(passed=False)
    item['DP_gate']=dp_gate
    for start in range(3):
        for factor in factors:
            ri=1 if factor==1. else 0 if factor<1. else 2
            rate=center*factor
            for chunk in chunks:
                c=copy.deepcopy(source);label='full' if chunk==source['length'] else str(chunk)
                c.update(name=f'ext_final_{model}_l{source["length"]}_c{label}_r{ri}_s{start}_v1',
                    phase='final_repeat',chunk=chunk,chunk_label=label,gpus=[2*start,2*start+1],seed=2026110107,
                    arrival_seed_base=20261131,verify_transfer=False,state_probe=False,nsys=False,
                    rate_index=ri,timeout=2400,extension_confirmation=True,
                    selection_scope='Independent confirmation of a recorded N24 screen. A negative screen remains a negative diagnostic.')
                c=finish(c,final_count(rate,start),rate=rate,start=start);c['P_slots']=source['P_slots']
                new.append(c);clean_names.append(c['name'])
            if dp_gate['passed']:
                c=copy.deepcopy(next(v for v in new if v['independent_start']==start and v['rate_index']==ri and v['chunk']==source['length']))
                c.update(name=c['name'].replace('_cfull_','_dp2_cfull_'),arm='DP2',P_slots=32)
                new.append(c);clean_names.append(c['name'])
    # Each profile has its own exact matched clean run. Overhead must remain
    # within the original 5% rule. A failed profile does not become clean time.
    for start in range(3):
        for chunk in chunks:
            clean=next(c for c in new if c['independent_start']==start and c['rate_index']==1 and c['chunk']==chunk and c['arm']=='PD_G1')
            c=copy.deepcopy(clean);c.update(name=clean['name'].replace('ext_final_','ext_profile_',1),phase='profile_control',
                nsys=True,nsys_graph_trace='graph',profile_clean_reference=clean['name'],timeout=3000)
            if model=='qwen8':c.update(nsys=False,nsys_P_only=True)
            new.append(c)
    if model=='qwen8':
        # Preserve a matched activity measurement before the next independent
        # center run. Do not finish every clean repeat before its first trace.
        paired=[]
        preferred=item['choice']['run']['chunk']
        ordered=sorted(new,key=lambda c:(c['chunk']!=preferred,c.get('independent_start',0)))
        for clean in ordered:
            if clean['phase']!='final_repeat' or clean['rate_index']!=1 or clean['arm']!='PD_G1':continue
            paired.append(clean)
            paired.extend(c for c in new if c.get('profile_clean_reference')==clean['name'])
        included={c['name'] for c in paired}
        new=paired+[c for c in new if c['name'] not in included]
    item.update(stage='FINAL_RUNS',finals=clean_names,profiles=[c['name'] for c in new if c['phase']=='profile_control'])
    return new


def advance():
    state=json.loads(PATH.read_text()) if PATH.exists() else {};runtime_path=ROOT/'state/extension_runtime_controls_20261102.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {};changed=False
    wire_path=ROOT/'state/extension_transport_20261102.json';wire=json.loads(wire_path.read_text()) if wire_path.exists() else {}
    slot_path=ROOT/'state/extension_slot_controls_20261102.json';slots=json.loads(slot_path.read_text()) if slot_path.exists() else {}
    hslot_path=ROOT/'state/extension_huginn_slot_controls_20261102.json';hslots=json.loads(hslot_path.read_text()) if hslot_path.exists() else {}
    long_path=ROOT/'state/extension_ouro_long_slot_controls_20261102.json';long_slots=json.loads(long_path.read_text()) if long_path.exists() else {}
    AR_path=ROOT/'state/ar_compiled_recovery_20261102.json';AR=json.loads(AR_path.read_text()) if AR_path.exists() else {}
    for model in ('ouro','huginn','qwen8'):
        families=[v for v in runtime.values() if v.get('model')==model]
        if model not in state:
            if model=='qwen8':
                if AR.get('stage') not in ('PILOTS_COMPLETE','SCREENING_PILOTS_COMPLETE'):continue
                families=[AR]
            else:
                if not families or any(v['stage'] not in ('PILOTS_COMPLETE','BLOCKED_GATE8','BLOCKED_COST_REFERENCE','UNSUPPORTED_OR_IGNORED_ASYNC_REQUEST') for v in families):continue
                if model not in wire or wire[model]['stage']!='PILOTS_COMPLETE':continue
                families=families+[wire[model]]
                if model=='ouro':
                    if slots.get('stage') not in ('PILOTS_COMPLETE','BLOCKED_NATIVE_D_REFERENCE'):continue
                    if long_slots.get('stage')!='PILOTS_COMPLETE':continue
                    families.extend((slots,long_slots))
                if model=='huginn':
                    if hslots.get('stage') not in ('PILOTS_COMPLETE','BLOCKED_NATIVE_D4_REFERENCE','NO_STRICT_RESERVE_AT_NEAR_P_SERVICE'):continue
                    families.append(hslots)
            options=[]
            for family in families:
                for name in family.get('pilots',[]):
                    if status(name)!='COMPLETE':continue
                    c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                    if c['arm']!='PD_G1':continue
                    screen=measured_screen(c)
                    if screen:options.append(screen)
            if not options:
                state[model]=dict(stage='NOT_RUN_NO_ELIGIBLE_PILOT',time=time.time());changed=True;continue
            write(ROOT/'state'/f'extension_confirmation_{model}_selection_20261102.json',dict(time=time.time(),all_screens=options,
                rule='Prefer clear D credit, Decode SLO margin, P backlog growth, and near-reference P service. Do not call this a GPU saturation proof.'))
            if model=='ouro':
                from . import extension_candidate_profiles
                import importlib
                importlib.reload(extension_candidate_profiles)
                choice=extension_candidate_profiles.choose(options)
                if choice is None:continue
            else:choice=max(options,key=lambda v:v['selection_score'])
            state[model]=qualify(model,choice);changed=True
        item=state[model]
        if item['stage']=='QUALIFICATION32' and terminal(item['qualification']):
            enqueue(final_jobs(model,item));changed=True
        if item['stage']=='FINAL_RUNS' and terminal(item['finals']+item['profiles']):
            item['stage']='CONFIRMATION_COMPLETE_PENDING_MARGIN';changed=True
        if item['stage'] in ('FINAL_RUNS','CONFIRMATION_COMPLETE_PENDING_MARGIN'):
            center=[name for name in item['finals'] if (lambda c:c.get('rate_index')==1 and c['arm']=='PD_G1')(
                json.loads((ROOT/'configs'/(name+'.json')).read_text()))]
            profiles=item['profiles']
            if model=='qwen8':
                preferred=item['choice']['run']['chunk']
                center=[name for name in center if json.loads((ROOT/'configs'/(name+'.json')).read_text())['chunk']==preferred]
                profiles=[name for name in profiles if json.loads((ROOT/'configs'/(name+'.json')).read_text())['chunk']==preferred]
            if not terminal(center+profiles):continue
            review=ROOT/'state'/f'extension_profile_review_{model}_20261102.json'
            if not review.exists() and not item.get('profile_review_pid'):
                import subprocess
                with (ROOT/'logs'/f'extension_profile_review_{model}_20261102.log').open('a') as log:
                    child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.extension_confirmation','--analyze-model',model],
                        cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item['profile_review_pid']=child.pid;changed=True
            elif review.exists():
                result=json.loads(review.read_text());item['profile_review']=result
                if item['stage']=='CONFIRMATION_COMPLETE_PENDING_MARGIN' and terminal(result.get('margin_diagnostics',[])):
                    item['stage']='CONFIRMATION_COMPLETE';changed=True
    if all(state.get(m,{}).get('stage') in ('CONFIRMATION_COMPLETE','BLOCKED_QUALIFICATION32','NOT_RUN_NO_ELIGIBLE_PILOT') for m in ('ouro','huginn')):
        write(ROOT/'state/extension_confirmation_terminal_20261102.json',dict(time=time.time(),models={m:state[m]['stage'] for m in ('ouro','huginn')}))
    if state.get('qwen8',{}).get('stage') in ('CONFIRMATION_COMPLETE','BLOCKED_QUALIFICATION32','NOT_RUN_NO_ELIGIBLE_PILOT'):
        write(ROOT/'state/ar_debug_terminal_20261102.json',dict(time=time.time(),stage=state['qwen8']['stage'],
            scope='A finished diagnostic is not necessarily a successful numerical or Prefill-heavy result. Preserve per-family verdicts.'))
    if changed:write(PATH,state)
    return state


def analyze_model(model):
    import fcntl
    from .profiles import parse
    item=json.loads(PATH.read_text())[model];analysis=[];new=[];selection={}
    # SQLite export and analysis can use substantial host resources. Do not
    # overlap this owned work with a clean GPU measurement.
    with (ROOT/'state/gpu_lease.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for name in item['profiles']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if status(name)!='COMPLETE':continue
            try:analysis.append(parse(ROOT/'raw'/name,cfg))
            except Exception as error:analysis.append(dict(name=name,status='BLOCKED_PROFILE_ANALYSIS',error=repr(error)))
    options={}
    for value in analysis:
        busy=next((v.get('kernel_busy_fraction') for v in value.get('devices',[]) if v['role']=='prefill'),None)
        if not value.get('eligible_profile_transfer_to_clean') or busy is None or busy<.8:continue
        cfg=json.loads((ROOT/'configs'/(value['clean_reference']+'.json')).read_text());screen=measured_screen(cfg)
        if not screen or not all(screen['checks'].values()):continue
        chunk=cfg['chunk']
        if chunk not in options or screen['run']['P_completion_rate_central']>options[chunk][1]['run']['P_completion_rate_central']:
            options[chunk]=(cfg,screen)
    strict_P_options=bool(options)
    if not options and model in ('ouro','huginn','qwen8'):
        # A negative P-compute gate does not answer the independent D-reserve
        # question. Run one small native diagnostic, not another Serving sweep.
        diagnostic=[]
        for name in item['finals']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if cfg.get('rate_index')!=1 or status(name)!='COMPLETE':continue
            screen=measured_screen(cfg)
            if screen and all(screen['checks'][key] for key in ('D_credit_clear','D_TPOT_margin_pass','D_independent_finite_reserve','P_backlog_growth')):
                diagnostic.append((cfg,screen))
        if diagnostic:
            cfg,screen=max(diagnostic,key=lambda pair:pair[1]['run']['P_completion_rate_central'])
            options[cfg['chunk']]=(cfg,screen)
    for chunk,(cfg,screen) in options.items():
        reference=json.loads((ROOT/'configs'/cfg['capacity_reference_file']).read_text())[model+'_L'+str(cfg['length'])]
        legal=32 if model=='qwen8' else 8 if model=='ouro' else 4 if cfg['length']==3500 and cfg.get('D_kv_blocks',0)>=28288 else 3
        count=min(legal,reference['D'][str(cfg['cap'])]['source']['batch'])
        rate=1.2*screen['run']['P_completion_rate_central'];source=cfg['name'];c=copy.deepcopy(cfg)
        c.update(name=f'ext_d_margin_{model}_c{chunk}_v1',phase='D_margin_diagnostic',count=count,
            prompt_ids=cfg['prompt_ids'][:count],request_caps=[cfg['cap']]*count,arrival_offsets=[0.]*count,
            prepared_D_arrival_offsets=[i/rate for i in range(count)],D_ready_pattern='FINITE_REGULAR_1P2X_OBSERVED_P_RATE',
            worker='ar_staggered_d' if model=='qwen8' else 'prepared_d_worker',prepare_D_only=model!='qwen8',
            prepared_release_P_on_each_import=model!='qwen8',
            P_slots=count,verify_transfer=model!='qwen8',state_probe=False,nsys=False,nsys_P_only=False,timeout=1800,
            reference_serving_run=source,diagnostic_ready_rate=rate)
        c['data_scope']=dict(c['data_scope'],offered_count=count)
        if model=='qwen8':c['gpus']=cfg['gpus'][-1:]
        purpose='STRICT_P_CANDIDATE_VALIDATION' if strict_P_options else 'D_RESERVE_ONLY_WITHOUT_QUALIFIED_P_COMPUTE'
        c['D_margin_selection_purpose']=purpose
        new.append(c);selection[c['name']]=dict(source=source,diagnostic=c['name'],count=count,rate=rate,purpose=purpose)
    write(ROOT/'state'/f'D_margin_selection_extension_{model}_20261102.json',selection);enqueue(new)
    write(ROOT/'state'/f'extension_profile_review_{model}_20261102.json',dict(time=time.time(),profiles=analysis,
        margin_diagnostics=[c['name'] for c in new],
        status=('MARGIN_DIAGNOSTICS_PENDING' if strict_P_options else 'D_ONLY_DIAGNOSTIC_PENDING_NO_QUALIFIED_P_COMPUTE') if new else 'NO_VALID_STRICT_P_SATURATION_SCREEN',
        scope='Only exact matched profiles inside the original 5% overhead bound can establish P kernel activity. No inferred GPU idle.'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--analyze-model');args=parser.parse_args()
    if args.analyze_model:
        try:analyze_model(args.analyze_model)
        except Exception as error:
            import traceback
            write(ROOT/'state'/f'extension_profile_review_{args.analyze_model}_20261102.json',dict(time=time.time(),
                status='BLOCKED_PROFILE_REVIEW',error=repr(error),traceback=traceback.format_exc(),margin_diagnostics=[],
                scope='No valid P saturation or D-margin claim follows from this failed CPU review.'))
            raise
    else:advance()
