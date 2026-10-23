"""Lock real arrivals and one primary objective before any FINAL outcome."""
import copy
import csv
import gzip
import json
import math
import random
from collections import defaultdict
import numpy as np
from .common import ROOT,PRIOR,GPUS,write,digest
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def arrivals(workload,rate,count,seed):
    rng=random.Random(seed);times=[];states=[];now=0.
    for index in range(count):
        if workload=='BURSTY':
            while True:
                high=int(now/8)%2==1
                local=rate*(2. if high else .5)
                proposed=now+rng.expovariate(local)
                boundary=(math.floor(now/8)+1)*8
                if proposed<boundary:
                    now=proposed;break
                now=boundary
            state='HIGH' if high else 'LOW'
        else:
            now+=rng.expovariate(rate);state='POISSON'
        times.append(now);states.append(state)
    if workload=='LONG8K_STRESS':
        # Preserve the causality probe of active short decodes followed by 8K.
        times=[0.]*7+[.25]+[max(.5,t) for t in times[8:]]
        states=['INITIAL_SHORT_BURST']*7+['DELAYED_8K']+states[8:]
    return times,states

def main():
    assert not (ROOT/'p3_final_workload_plan.json').exists(),'Preserve frozen FINAL traffic'
    selected=json.loads((ROOT/'objective_shortlist.json').read_text())
    primary=selected['primary_objective'];family=primary['objective']
    developmental=json.loads((ROOT/'objective_shortlist_development.json').read_text())['arms']
    validation=list(csv.DictReader((ROOT/'serving_results.csv').open()))
    validation=[r for r in validation if r['split']=='VALIDATION']
    worst_cap_fraction=max(int(r['caps'])/max(1,int(r['offered'])) for r in validation)
    final_cap=4096 if worst_cap_fraction>.05 else 1024
    write(ROOT/'final_output_cap_amendment.json',dict(initial_pilot_cap=1024,final_cap=final_cap,
        worst_validation_cohort_cap_fraction=worst_cap_fraction,trigger=.05,
        rule='Before FINAL, raise every arm to4096 if any VALIDATION cohort has more than5% capped outputs. No old score is relabeled.',
        reason='Completion-oriented natural generation. Remaining caps still fail complete-response attainment.'))
    arms=[dict(c) for c in developmental if c['label'] in (family+'_J',family+'_G',family+'_G1024',family+'_C')]
    atomic=dict(primary,label=family+'_CA',runtime='atomic',surface='C_ONLY');arms.append(atomic)
    fixed=dict(selected['best_tested_fixed'],label='BEST_FIXED');arms.append(fixed)
    mixed_C=selected['best_tested_mixed']['C']
    arms.extend([dict(label='ATOMIC',runtime='atomic',C=1024,fairness='stock'),
        dict(label='MIXED_STOCK',runtime='mixed',C=mixed_C,fairness='stock'),
        dict(label='MIXED_FAIR',runtime='mixed',C=mixed_C,fairness='completion')])
    cases=json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())['cases']
    final=[c for c in cases if c['split']=='FINAL']
    byfamily=defaultdict(list)
    for c in final:byfamily[c['family']].append(c)
    chats=sorted(f for f,items in byfamily.items() if items[0]['domain']=='chat')
    articles=sorted(set(byfamily)-set(chats))
    quiet_families=set(chats[len(chats)//2:]+articles[len(articles)//2:])
    pools={}
    for lane,quiet in [('CHARACTERIZATION',False),('QUIET',True)]:
        for domain in ('chat','long','stress8k'):
            pools[(lane,domain)]=[c for c in final if c['domain']==domain and (c['family'] in quiet_families)==quiet]
    assert all(pools[k] for k in pools)
    scenarios=[];arrival_rows=[]
    workloads=['CHAT','MIXED','LONG','BURSTY','LONG8K_STRESS']
    def scenario(workload,rate,start,lane):
        n=12;seed=2026102400+100*workloads.index(workload)+10*start+int(rate*10)
        rng=random.Random(seed)
        if workload=='CHAT':domains=['chat']*n
        elif workload=='LONG':domains=['long']*n
        elif workload=='LONG8K_STRESS':domains=['chat']*7+['stress8k']+['chat']*4
        else:
            domains=['chat']*9+['long']*3;rng.shuffle(domains)
        cursors=defaultdict(int);chosen=[]
        for domain in domains:
            pool=pools[(lane,domain)]
            index=(start*12+cursors[domain])%len(pool);cursors[domain]+=1
            chosen.append(pool[index]['id'])
        times,states=arrivals(workload,rate,n,seed)
        name=f'FINAL_{lane}_{workload}_R{rate:g}_S{start}'
        record=dict(id=name,split='FINAL',lane=lane,workload=workload,rate=rate,seed=seed,
            case_ids=chosen,arrivals_s=times,duration_s=max(times)+.001,
            realized_offered_rate_rps=n/(max(times)+.001),
            rate_parameter_scope='Poisson rate for ordinary traffic. Base intensity scale for BURSTY and hybrid8K.',
            low_state_rate_rps=.5*rate if workload=='BURSTY' else None,
            high_state_rate_rps=2.*rate if workload=='BURSTY' else None,
            arrival_type='TWO_STATE_MODULATED_POISSON' if workload=='BURSTY' else 'BURST_PLUS_DELAYED_8K' if workload=='LONG8K_STRESS' else 'POISSON',
            burst_dwell_s=8 if workload=='BURSTY' else None,
            burst_high_low_ratio=4 if workload=='BURSTY' else None,
            qualification='Twelve offered natural requests, not qualified sustainable capacity.')
        scenarios.append(record)
        for index,(case,time,state) in enumerate(zip(chosen,times,states)):
            arrival_rows.append(dict(scenario=name,index=index,case_id=case,arrival_s=time,state=state,seed=seed))
        return name
    manifest=ROOT/'final_serving_scenarios.json'
    jobs=[];quiet_jobs=[]
    for w_i,workload in enumerate(workloads):
        for rate_i,rate in enumerate((.25,.9)):
            for start in range(3):
                gpu=GPUS[(w_i+rate_i+start)%4]
                name=scenario(workload,rate,start,'CHARACTERIZATION')
                ordered=arms[(w_i+rate_i+start)%len(arms):]+arms[:(w_i+rate_i+start)%len(arms)]
                if start%2:ordered=list(reversed(ordered))
                jobs.append(dict(id=f'dual_p3_CHAR_{workload}_R{rate:g}_S{start}_gpu{gpu}_v1',
                    kind='dual_serving',gpu=gpu,width=4 if workload=='CHAT' and rate==.25 else 8,cells=ordered,
                    scenarios=[name],scenario_manifest=manifest.name,cap=final_cap,budget=2048,blocks=18432,
                    graphs=True,streams=True,drain_s=600,timeout_s=10800,
                    lane='LIVE_MEASURED_FINAL_CHARACTERIZATION_FOUR_WORKERS',
                    normalization_sha256=digest(ROOT/'objective_normalization_v1.json'),workload_lock_sha256=digest(ROOT/'workload_lock.json')))
    for w_i,workload in enumerate(('MIXED','LONG8K_STRESS')):
        for start in range(3):
            gpu=GPUS[(w_i+start)%4];name=scenario(workload,.9,start,'QUIET')
            keep=(family+'_J',family+'_G',family+'_G1024',family+'_C','BEST_FIXED','MIXED_STOCK','MIXED_FAIR')
            ordered=[c for c in arms if c['label'] in keep]
            ordered=ordered[start:]+ordered[:start]
            if start%2:ordered=list(reversed(ordered))
            quiet_jobs.append(dict(id=f'dual_p3_QUIET_{workload}_S{start}_gpu{gpu}_v1',
                kind='dual_serving',gpu=gpu,width=8,cells=ordered,scenarios=[name],scenario_manifest=manifest.name,
                cap=final_cap,budget=2048,blocks=18432,graphs=True,streams=True,drain_s=600,timeout_s=10800,
                lane='LIVE_MEASURED_SINGLE_MEASUREMENT_WORKER_FIXED_GPU0_3_BACKGROUND',
                normalization_sha256=digest(ROOT/'objective_normalization_v1.json'),workload_lock_sha256=digest(ROOT/'workload_lock.json')))
    write(manifest,dict(scenarios=scenarios))
    for job in jobs+quiet_jobs:job['scenario_sha256']=digest(manifest)
    csv_file(ROOT/'arrival_manifest.csv',arrival_rows)
    write(ROOT/'p3_characterization_plan.json',dict(gpus=GPUS,jobs=jobs,
        scope='Four-worker FINAL characterization. Same-GPU complete policy crossover per natural arrival cohort.'))
    write(ROOT/'p3_quiet_confirmation_plan.json',dict(gpus=GPUS,groups=[dict(jobs=[j]) for j in quiet_jobs],
        scope='All authorized GPUs are isolation-reserved. One measurement worker;0–3 background untouched. QUIET families never enter characterization.'))
    subplans=[]
    if (ROOT/'p2_alternative_diagnostic_plan.json').exists():subplans.append('p2_alternative_diagnostic_plan.json')
    # Separate ready-stream event timing from the clean serving lanes.
    calibration={domain:[c for c in cases if c['split']=='CALIBRATION' and c['domain']==domain] for domain in ('chat','long','stress8k')}
    audit_scenarios=[];audit_jobs=[]
    for i,gpu in enumerate(GPUS):
        width=4 if i<2 else 8;B=40. if i%2==0 else 120.;domain='long' if width==4 else 'stress8k'
        offered=calibration['chat'][:width-1]+[calibration[domain][i%len(calibration[domain])]]
        sid=f'CAL_EVENT_N{width}_B{int(B)}'
        audit_scenarios.append(dict(id=sid,split='CALIBRATION',workload=domain.upper()+'_EVENT_AUDIT',
            case_ids=[c['id'] for c in offered],arrivals_s=[0.]*(width-1)+[.25],duration_s=.251))
        audit_cells=[]
        for c in arms:
            if c['label'] not in (family+'_J',family+'_G',family+'_C','MIXED_FAIR'):continue
            for observed in (False,True):
                copy_cell=dict(c,label=c['label']+('_OBS' if observed else '_NOOBS'),observe=observed)
                if copy_cell.get('surface'):copy_cell['budget_ms']=B
                audit_cells.append(copy_cell)
        if i%2:audit_cells=list(reversed(audit_cells))
        audit_jobs.append(dict(id=f'dual_p2_EVENT_N{width}_B{int(B)}_gpu{gpu}_v1',kind='dual_serving',gpu=gpu,
            width=width,cells=audit_cells,scenarios=[sid],scenario_manifest='event_audit_scenarios.json',cap=1024,
            budget=2048,blocks=18432,graphs=True,streams=True,drain_s=600,timeout_s=7200,
            lane='LIVE_MEASURED_OBSERVER_PARITY_AND_BUDGET_AUDIT_NOT_CLEAN_SERVING',
            normalization_sha256=digest(ROOT/'objective_normalization_v1.json'),workload_lock_sha256=digest(ROOT/'workload_lock.json')))
    audit_manifest=ROOT/'event_audit_scenarios.json';write(audit_manifest,dict(scenarios=audit_scenarios))
    for j in audit_jobs:j['scenario_sha256']=digest(audit_manifest)
    write(ROOT/'p2_event_audit_plan.json',dict(gpus=GPUS,jobs=audit_jobs))
    subplans.append('p2_event_audit_plan.json')
    subplans+=['p3_characterization_plan.json','p3_quiet_confirmation_plan.json']
    write(ROOT/'p3_final_workload_plan.json',dict(gpus=GPUS,subplans=subplans))
    # The quality question set is historical and fixed, not a new blind accuracy set.
    old_manifest=PRIOR/'quality_manifest.json';quality=json.loads(old_manifest.read_text())
    write(ROOT/'quality_manifest.json',dict(quality,source_manifest=str(old_manifest),source_sha256=digest(old_manifest),
        scope='Same public 64 GSM8K and50 MATH questions. New matched configurations, not a new blind dataset.'))
    quality_arms=[dict(label='F0A',runtime='atomic',C=1024,fairness='stock')]
    quality_arms += [dict(c) for c in arms if c['label'] in (family+'_J',family+'_G',family+'_G1024',family+'_C',family+'_CA','BEST_FIXED','MIXED_STOCK','MIXED_FAIR')]
    for c in quality_arms:
        if c.get('surface'):c['optimizer']=c['label']
    quality_jobs=[]
    cost=PRIOR/'cost_model_artifacts/service_lookup_v1.json'
    for index,gpu in enumerate(GPUS):
        ordered=quality_arms[index:]+quality_arms[:index]
        quality_jobs.append(dict(id=f'dual_p4_QUALITY_gpu{gpu}_v1',kind='dual_quality',gpu=gpu,cells=ordered,
            ids=[r['id'] for i,r in enumerate(quality['rows']) if i%4==index],width=4,cap=4096,
            compute_budget_ms=120.,budget=2048,blocks=18432,cost_model=str(cost),cost_model_sha256=digest(cost),timeout_s=21600))
    write(ROOT/'p4_quality_plan.json',dict(gpus=GPUS,jobs=quality_jobs,
        scope='Nine-arm full-answer public quality pilot,114 questions per arm. Not clean serving timing.'))
    baseline=list(csv.DictReader((ROOT/'contention_raw.csv').open()))
    reference={str(P):float(np.median([float(r['median_ITL_ms']) for r in baseline if r['workers']=='1' and int(r['P'])==P])) for P in (2048,8192)}
    write(ROOT/'slo_lock.json',dict(primary=dict(TTFT_ms=5000,every_gap_TBT_ms=200),
        absolute_TTFT_ms=[2000,5000,10000],absolute_TBT_ms=[100,150,200,250,500],
        decode_reference=reference,reference_scope='Median of24 warmed cohort-median gaps per context after all first outputs. Not pooled token P50.',
        multiplier_tiers={k:[5*v,10*v,25*v] for k,v in reference.items()},
        capacity_offered_minimum=1000,capacity_minimum_starts=3,capacity_target=.90,
        current_finite_cohort_offered=12,source_role='Frozen before FINAL serving and quality outcomes.'))
    write(ROOT/'final_freeze.json',dict(objective_sha256=digest(ROOT/'objective_shortlist.json'),
        scenario_sha256=digest(manifest),slo_sha256=digest(ROOT/'slo_lock.json'),
        quiet_families=sorted(quiet_families),selector_runtime_sha256=digest(ROOT.parent.parent/'ouro_dual_objective_cg/adaptive_selector.py')
            if (ROOT.parent.parent/'ouro_dual_objective_cg/adaptive_selector.py').exists() else None,
        workloads=workloads,characterization_jobs=len(jobs),quiet_jobs=len(quiet_jobs),quality_generations=len(quality['rows'])*len(quality_arms),
        external_SLO_safety_claim=False))
    print('Frozen FINAL:',len(jobs),'characterization jobs,',len(quiet_jobs),'quiet jobs,',len(quality['rows'])*len(quality_arms),'quality generations')

if __name__=='__main__':main()
