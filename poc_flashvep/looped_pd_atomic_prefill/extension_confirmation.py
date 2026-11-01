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
    for label,chunk,factor in [('reference',source['length'],.7),('repeat',source['length'],1.2),('tuned',tuned,1.)]:
        cfg=copy.deepcopy(source);cfg.update(name=f'ext_confirm32_{model}_{label}_v1',phase='final_correctness',
            chunk=chunk,chunk_label='full' if chunk==source['length'] else str(chunk),gpus=[0,1] if model=='ouro' else [4,5],
            verify_transfer=False,state_probe=False,nsys=False,timeout=1800)
        cfg=finish(cfg,32,rate=source['arrival_rate']*factor)
        cfg.update(dataset=dataset,prompt_ids=[r['prompt_id'] for r in subset],
            data_scope=dict(original_pool_records=32,offered_count=32,explicit_cyclic_reuse=False),
            qualification_scope='Independent prompt subset, arrival seed, and Sampling seed. Existing native valid-KV and hidden-byte gates remain separate.')
        output.append(cfg)
    enqueue(output)
    return dict(stage='QUALIFICATION32',choice=choice,source=source,tuned_chunk=tuned,qualification=[c['name'] for c in output])


def final_jobs(model,item):
    ref,rep,tuned=item['qualification']
    gate=compare(ref,rep) if status(ref)==status(rep)=='COMPLETE' else dict(passed=False)
    chunk_gate=compare(ref,tuned) if status(ref)==status(tuned)=='COMPLETE' else dict(passed=False)
    if gate['passed']:
        gate['work']=full_work(ref);gate['atomic']=atomic(ref)
        gate['passed'] &= gate['work']['passed'] and gate['atomic']['passed']
    if chunk_gate['passed']:
        chunk_gate['work']=full_work(tuned);chunk_gate['passed'] &= chunk_gate['work']['passed']
    item.update(repeat_gate=gate,tuned_gate=chunk_gate)
    if not gate['passed']:item['stage']='BLOCKED_QUALIFICATION32';return []
    source=item['source'];center=source['arrival_rate'];new=[];clean_names=[]
    chunks=list(dict.fromkeys([source['length']]+([item['tuned_chunk']] if chunk_gate['passed'] else [])))
    for start in range(3):
        # Three matched rates retain both sides of the selected pressure cell.
        for ri,factor in enumerate((.8,1.,1.2)):
            rate=center*factor
            for chunk in chunks:
                c=copy.deepcopy(source);label='full' if chunk==source['length'] else str(chunk)
                c.update(name=f'ext_final_{model}_l{source["length"]}_c{label}_r{ri}_s{start}_v1',
                    phase='final_repeat',chunk=chunk,chunk_label=label,gpus=[2*start,2*start+1],seed=2026110107,
                    arrival_seed_base=20261131,verify_transfer=False,state_probe=False,nsys=False,
                    rate_index=ri,timeout=2400,extension_confirmation=True,
                    selection_scope='Independent confirmation of a recorded N24 screen. A negative screen remains a negative diagnostic.')
                c=finish(c,final_count(rate,start),rate=rate,start=start);new.append(c);clean_names.append(c['name'])
    # Each profile has its own exact matched clean run. Overhead must remain
    # within the original 5% rule. A failed profile does not become clean time.
    for start in range(3):
        for chunk in chunks:
            clean=next(c for c in new if c['independent_start']==start and c['rate_index']==1 and c['chunk']==chunk)
            c=copy.deepcopy(clean);c.update(name=clean['name'].replace('ext_final_','ext_profile_',1),phase='profile_control',
                nsys=True,nsys_graph_trace='graph',profile_clean_reference=clean['name'],timeout=3000)
            new.append(c)
    item.update(stage='FINAL_RUNS',finals=clean_names,profiles=[c['name'] for c in new if c['phase']=='profile_control'])
    return new


def advance():
    state=json.loads(PATH.read_text()) if PATH.exists() else {};runtime_path=ROOT/'state/extension_runtime_controls_20261102.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {};changed=False
    wire_path=ROOT/'state/extension_transport_20261102.json';wire=json.loads(wire_path.read_text()) if wire_path.exists() else {}
    for model in ('ouro','huginn'):
        families=[v for v in runtime.values() if v.get('model')==model]
        if model not in state:
            if not families or any(v['stage'] not in ('PILOTS_COMPLETE','BLOCKED_GATE8','BLOCKED_COST_REFERENCE') for v in families):continue
            if model not in wire or wire[model]['stage']!='PILOTS_COMPLETE':continue
            options=[]
            for family in families+[wire[model]]:
                for name in family.get('pilots',[]):
                    if status(name)!='COMPLETE':continue
                    c=json.loads((ROOT/'configs'/(name+'.json')).read_text());screen=measured_screen(c)
                    if screen:options.append(screen)
            if not options:
                state[model]=dict(stage='NOT_RUN_NO_ELIGIBLE_PILOT',time=time.time());changed=True;continue
            write(ROOT/'state'/f'extension_confirmation_{model}_selection_20261102.json',dict(time=time.time(),all_screens=options,
                rule='Prefer clear D credit, Decode SLO margin, P backlog growth, and near-reference P service. Do not call this a GPU saturation proof.'))
            state[model]=qualify(model,max(options,key=lambda v:v['selection_score']));changed=True
        item=state[model]
        if item['stage']=='QUALIFICATION32' and terminal(item['qualification']):
            enqueue(final_jobs(model,item));changed=True
        if item['stage']=='FINAL_RUNS' and terminal(item['finals']+item['profiles']):
            item['stage']='CONFIRMATION_COMPLETE_PENDING_MARGIN';changed=True
    if changed:write(PATH,state)
    return state


if __name__=='__main__':advance()
