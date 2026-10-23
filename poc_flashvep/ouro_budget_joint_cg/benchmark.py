"""Preserve native kernels. Separate correctness, service, and clean serving."""
import gc
import json
import random
import time
import torch
from .common import ROOT,MODEL,write,config,guard
from .fairness import attach
from .actions import ChunkActions,scripted,cycling
from .service_observer import ServiceObserver

def engine(model,mode,C,width,job):
    from vllm_rlt.engine.stage_engine import StageEngine
    return StageEngine(model,**config(mode,C,width,blocks=job.get('blocks',18432),budget=job.get('budget',2048),
        graphs=job.get('graphs',True),streams=job.get('streams',True)))

def select_cases(manifest,split,P,N,trial):
    long=[c for c in manifest['cases'] if c['split']==split and c.get('P')==P]
    short=[c for c in manifest['cases'] if c['split']==split and c['domain']=='conversation_proxy' and len(c['input_ids'])<=1024]
    assert long and len(short)>=N-1,(split,P,N)
    cases=[short[(trial*(N-1)+i)%len(short)] for i in range(N-1)]+[long[trial%len(long)]]
    return cases,[0.]*(N-1)+[.250 if N>1 else 0.]

@torch.inference_mode()
def numerical(model,job,target):
    from transformers import AutoTokenizer
    from vllm_rlt import SamplingParams
    from vllm_rlt.request import RequestPhase
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    fixture=tokenizer.encode('A rectangle has length twelve and width seven. Explain why its area is eighty-four. '*2000)
    results=[]
    for P in job['lengths']:
        reference=None
        for mode in job.get('modes',('atomic','grouped','grouped3','scripted')):
            e=engine(model,'split' if mode in ('scripted','cycled') else mode,2048 if mode in ('atomic','scripted','cycled') else 512,2,
                dict(job,blocks=max(512,4*((P+32+15)//16))))
            fairness=attach(e,True)
            hook=ChunkActions(e,scripted if mode=='scripted' else cycling) if mode in ('scripted','cycled') else None
            observer=ServiceObserver(e,dict(run_id=job['id'],policy=mode,P=P,physical_GPU=guard(),lane='PRETRAINED_NUMERICAL'))
            e.add_request('probe',fixture[:P],SamplingParams(max_tokens=8,max_loops=4,exit_threshold=1.,temperature=0))
            hidden=logits=None;kv=[];final=None
            try:
                while e.has_unfinished_requests():
                    request=e.scheduler.requests.get('probe')
                    if request and request.phase==RequestPhase.CODA and hidden is None:
                        e.model_runner.synchronize()
                        hidden=request.hidden_state.clone().cpu()
                        logits=model.coda(request.hidden_state.reshape(1,-1)).cpu()
                        for depth in range(4):
                            for layer in range(24):
                                k,v=e.cache_manager.read(layer,'probe',depth,P)
                                kv.extend((k.cpu(),v.cpu()))
                    for output in e.step():
                        final=output
                quanta=observer.finish()
                work=sum(r['prompt_token_loop_work'] for r in quanta)
                assert work==4*P and all(d==4 for d in final.exit_depths)
                row=dict(mode=mode,P=P,tokens=list(final.token_ids),depths=list(final.exit_depths),
                    exact_prompt_work=work,expected_prompt_work=4*P,
                    all_submissions_have_service_events=all(r['service_ms']>=0 for r in quanta),
                    decisions=[] if hook is None else hook.decisions,service_observations=quanta)
                if reference is None:
                    reference=(list(final.token_ids),hidden,logits,kv)
                else:
                    tokens,h,l,values=reference
                    def difference(a,b):
                        if torch.equal(a,b):
                            return dict(exact=True,max_abs=0.,relative_L2=0.)
                        delta=a.float()-b.float()
                        return dict(exact=False,max_abs=float(delta.abs().max()),relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.float()).clamp_min(1e-12)))
                    row['comparison']=dict(tokens_exact=tokens==list(final.token_ids),hidden=difference(h,hidden),logits=difference(l,logits),
                        KV_exact=all(torch.equal(a,b) for a,b in zip(values,kv)),
                        KV_max_relative_L2=max(difference(a,b)['relative_L2'] for a,b in zip(values,kv)))
                    row['numerical_tolerance_pass']=max(row['comparison']['hidden']['relative_L2'],row['comparison']['logits']['relative_L2'],row['comparison']['KV_max_relative_L2'])<=.03
                write(target/f'P{P}_{mode}.json',row)
                results.append({k:v for k,v in row.items() if k not in ('service_observations','decisions')})
            finally:
                observer.close()
                if hook:
                    hook.close()
                e.close();del e,observer,hidden,logits,kv,fairness,hook
                gc.collect();torch.cuda.empty_cache()
        del reference
    return dict(integrity_pass=True,numerical_tolerance_pass=all(r.get('numerical_tolerance_pass',True) for r in results),
        results=results,scope='Pretrained isolated probes. Not full-answer quality.')

@torch.inference_mode()
def fixed_service(model,job,target):
    from ..ouro_chunk_loop_slo import worker as frontend
    frontend.ROOT=ROOT
    manifest=json.loads((ROOT/'source_family_manifest.json').read_text())
    reports=[]
    for cell in job['cells']:
        mode,C=cell['mode'],cell['C'];N=job['shape'][1];P=job['shape'][0]
        e=engine(model,mode,C,N,job);e.frontend_stop_token_ids=manifest['stop_ids']
        fairness=attach(e,cell.get('fairness','stock')=='completion')
        cases,arrivals=select_cases(manifest,job['split'],P,N,job['trials'][0])
        frontend.serve(e,cases,16,arrivals,.251,observe=False)
        for trial in job['trials']:
            fairness_begin=len(fairness.events)
            cases,arrivals=select_cases(manifest,job['split'],P,N,trial)
            meta=dict(run_id=job['id'],policy=mode,C=C,configured_N=N,P=P,split=job['split'],trial=trial,
                physical_GPU=guard(),fairness=cell.get('fairness','stock'),families=[c['family'] for c in cases])
            observer=ServiceObserver(e,meta)
            torch.cuda.reset_peak_memory_stats()
            try:
                data=frontend.serve(e,cases,job.get('cap',64),arrivals,.251,observe=False)
                quanta=observer.finish()
                for r in data['requests']:
                    work=sum(s['tokens']*s['G'] for q in quanta for s in q['spans'] if s['id']==r['id'] and s['phase'] in ('prefill','prefilling_recurrent'))
                    assert work==4*r['input_tokens'],(r['id'],work,r['input_tokens'])
                    assert all(d==4 for d in r['depths'])
                data.update(meta,service_observations=quanta,fairness_events=list(fairness.events[fairness_begin:]),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    lane='MEASURED_OBSERVER' if job.get('quiet') else 'EXPLORATORY_SHARED_HOST',
                    full_R4_prompt_accounting=True)
                name=f'{mode}_C{C}_{meta["fairness"]}_T{trial}.json'
                write(target/name,data)
                reports.append(dict(mode=mode,C=C,trial=trial,fairness=meta['fairness'],quantums=len(quanta)))
            finally:
                observer.close();del observer
        e.close();del e,fairness,data
        gc.collect();torch.cuda.empty_cache()
    return dict(completed=True,cohorts=reports)

@torch.inference_mode()
def live_service(model,job,target):
    from ..ouro_chunk_loop_slo import worker as frontend
    from .cost_predictor import ServiceLookup
    from .online import OnlineControl
    frontend.ROOT=ROOT
    manifest=json.loads((ROOT/'source_family_manifest.json').read_text())
    artifact=ROOT/job['cost_model']
    assert __import__('hashlib').sha256(artifact.read_bytes()).hexdigest()==job['cost_model_sha256']
    lookup=ServiceLookup(json.loads(artifact.read_text()))
    reports=[]
    for cell in job['cells']:
        P,N=job['shape'];mode=cell['runtime'];C=cell.get('C',2048)
        e=engine(model,mode,C,N,job);e.frontend_stop_token_ids=manifest['stop_ids']
        fairness=attach(e,cell.get('fairness','completion')=='completion')
        control=None
        if cell.get('optimizer'):
            control=OnlineControl(e,lookup,cell['optimizer'],cell['budget_ms'],
                cell.get('fixed_C',1024),cell.get('objective','U2'),cell.get('yield_penalty_ms',0.),
                cell.get('feedback','FIXED'),cell.get('target_ITL_ms',270.))
        cases,arrivals=select_cases(manifest,job['split'],P,N,job['trials'][0])
        frontend.serve(e,cases,16,arrivals,.251,observe=False)
        for trial in job['trials']:
            if control:
                control.reset_cohort()
            fairness_begin=len(fairness.events)
            cases,arrivals=select_cases(manifest,job['split'],P,N,trial)
            meta=dict(run_id=job['id'],label=cell['label'],policy=mode,C=C,configured_N=N,P=P,
                split=job['split'],trial=trial,physical_GPU=guard(),fairness=cell.get('fairness','completion'),
                families=[c['family'] for c in cases],configured_budget_ms=cell.get('budget_ms'))
            observer=ServiceObserver(e,meta,prefill_only=True) if job.get('observe',False) else None
            torch.cuda.reset_peak_memory_stats()
            try:
                before=time.perf_counter()
                data=frontend.serve(e,cases,job.get('cap',128),arrivals,.251,observe=False)
                rows=[] if observer is None else observer.finish()
                if observer:
                    for r in data['requests']:
                        work=sum(s['tokens']*s['G'] for q in rows for s in q['spans']
                            if s['id']==r['id'] and s['phase'] in ('prefill','prefilling_recurrent'))
                        assert work==4*r['input_tokens'],(r['id'],work,r['input_tokens'])
                assert all(d==4 for r in data['requests'] for d in r['depths'])
                data.update(meta,service_observations=rows,cell=cell,cohort_wall_s=time.perf_counter()-before,
                    fairness_events=list(fairness.events[fairness_begin:]),
                    optimizer_decisions=[] if control is None else list(control.selector.trace),
                    chunk_decisions=[] if control is None else list(control.actions.decisions),
                    budget_admission=[] if control is None else list(control.admission.trace),
                    output_control_events=[] if control is None else list(control.output_events),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    lane=('MEASURED_OBSERVER' if observer else 'MEASURED_CLEAN') if job.get('quiet') else 'EXPLORATORY_SHARED_HOST',
                    cost_model_status='EXPERIMENTAL_BUDGET_RISK',
                    prompt_work_accounting='MEASURED_ALL_PREFILLS' if observer else 'Previously qualified full-R4 native path. No heavy per-dispatch observer.',
                    request_packing='One prefill span per adaptive batch. Native fixed/Mixed retain native packing.',
                    stopping='Natural greedy EOS allowed. Bounded pilot is not complete-answer quality or capacity.')
                write(target/f'{cell["label"]}_T{trial}.json',data)
                reports.append(dict(label=cell['label'],trial=trial,requests=len(data['requests']),lane=data['lane']))
            finally:
                if observer:
                    observer.close()
        if control:
            control.close()
        e.close();del e,fairness,control,data
        gc.collect();torch.cuda.empty_cache()
    return dict(completed=True,cohorts=reports)

def run_job(model,job,target):
    if job['kind']=='numerical':
        return numerical(model,job,target)
    if job['kind'] in ('service','fairness'):
        return fixed_service(model,job,target)
    if job['kind']=='live_budget':
        return live_service(model,job,target)
    if job['kind']=='observer_control':
        from .measurement_controls import run_job as controls
        return controls(model,job,target)
    if job['kind']=='isolated_service':
        from .isolated_service import run_job as isolated
        return isolated(model,job,target)
    raise ValueError(job['kind'])
