"""Matched full-answer greedy pilot. No forced tokens or early model exit."""
import argparse
import gc
import json
import time
import torch
from .common import ROOT,PRIOR,MODEL,write,digest
from .benchmark import engine
from .fairness import attach
from .online import OnlineControl
from .cost_predictor import ServiceLookup

def register(budget):
    old=PRIOR/'quality_manifest.json'
    previous=json.loads(old.read_text())
    target=ROOT/'quality_manifest.json';assert not target.exists()
    write(target,dict(previous,source_manifest=str(old),source_manifest_sha256=digest(old),
        scope='Matched historical 64 GSM8K / 50 MATH public pilot. Not a new blind accuracy set.',
        budget_ms=budget,model_revision=MODEL.name))
    selection=json.loads((ROOT/'pilot_training_selection.json').read_text())
    mixed=next(c['C'] for c in selection['choices'] if c['P']==2033 and c['mode']=='mixed')
    fixedG=next(c['fixed_C'] for j in json.loads((ROOT/'p3_budget_pilot_plan.json').read_text())['jobs']
        if j['shape'][0]==2033 for c in j['cells'] if c['label']=='G1' and c['budget_ms']==budget)
    artifact=ROOT/'cost_model_artifacts/service_lookup_v1.json'
    jobs=[]
    for gpu in range(8):
        cells=[dict(label='C1A',runtime='atomic',optimizer='C_ONLY',C=2048),
            dict(label='C1S',runtime='split',optimizer='C_ONLY',C=2048),
            dict(label='G1',runtime='split',optimizer='G_ONLY',C=2048,fixed_C=fixedG),
            dict(label='J1',runtime='split',optimizer='JOINT',C=2048),
            dict(label='M1',runtime='mixed',C=mixed),
            dict(label='M0',runtime='mixed',C=mixed,fairness='stock'),
            dict(label='F0A',runtime='atomic',C=1024,fairness='stock')]
        cells=cells[gpu%len(cells):]+cells[:gpu%len(cells)]
        jobs.append(dict(id=f'cb_p6_quality_B{budget}_gpu{gpu}_v1',kind='quality',gpu=gpu,cells=cells,
            ids=[r['id'] for i,r in enumerate(previous['rows']) if i%8==gpu],width=4,cap=4096,
            compute_budget_ms=budget,budget=2048,cost_model=str(artifact.relative_to(ROOT)),
            cost_model_sha256=digest(artifact),timeout_s=14400))
    plan=ROOT/'p6_quality_plan.json';assert not plan.exists()
    write(plan,dict(gpus=list(range(8)),jobs=jobs,
        scope='Full-answer matched natural accuracy pilot. Quality timing includes detokenization and scoring-related text checks. Not clean capacity.'))

@torch.inference_mode()
def run_job(model,job,target):
    from transformers import AutoTokenizer
    from vllm_rlt import SamplingParams
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    manifest=json.loads((ROOT/'quality_manifest.json').read_text())
    cases=[r for r in manifest['rows'] if r['id'] in job['ids']]
    artifact=ROOT/job['cost_model'];assert digest(artifact)==job['cost_model_sha256']
    lookup=ServiceLookup(json.loads(artifact.read_text()));results=[]
    for cell in job['cells']:
        label=cell['label'];e=engine(model,cell['runtime'],cell['C'],job['width'],job)
        attach(e,cell.get('fairness','completion')=='completion')
        control=None
        if cell.get('optimizer'):
            control=OnlineControl(e,lookup,cell['optimizer'],job['compute_budget_ms'],cell.get('fixed_C',1024))
        e.add_request('warm',tokenizer.encode('Q: What is two plus two?\nA:'),
            SamplingParams(max_tokens=8,max_loops=4,exit_threshold=1.,temperature=0))
        while e.has_unfinished_requests():
            e.step()
        for offset in range(0,len(cases),job['width']):
            group=cases[offset:offset+job['width']];records={};byid={r['id']:r for r in group}
            if control:
                control.reset_cohort()
            for case in group:
                rid=case['id'];now=time.perf_counter()
                records[rid]=dict(id=rid,dataset=case['dataset'],prompt=case['prompt'],prompt_ids=case['input_ids'],
                    gold=case['answer'],policy=label,cap=job['cap'],output_ids=[],token_times=[],depths=[],start=now,
                    evidence='LIVE_FULL_ANSWER_GREEDY_QUALITY_PILOT_NOT_CLEAN_SERVING')
                e.add_request(rid,case['input_ids'],SamplingParams(max_tokens=job['cap'],max_loops=4,exit_threshold=1.,temperature=0))
            while e.has_unfinished_requests():
                for output in e.step():
                    now=time.perf_counter();r=records[output.request_id];ids=list(output.token_ids)
                    if len(ids)>len(r['output_ids']):
                        assert len(ids)==len(r['output_ids'])+1
                        r['output_ids']=ids;r['depths']=list(output.exit_depths);r['token_times'].append(now)
                    text=tokenizer.decode(ids,skip_special_tokens=False)
                    stop=next((s for s in byid[output.request_id]['stop_strings'] if s in text),None)
                    if output.finished or stop is not None:
                        reason=getattr(output.finish_reason,'value',output.finish_reason) if output.finished else 'frontend_registered_stop'
                        if not output.finished:
                            e.abort_request(output.request_id)
                        r.update(text=text,finish_reason=reason,registered_stop=stop,end=now,output_tokens=len(ids),
                            full_R4=all(d==4 for d in r['depths']),E2E_s=now-r['start'])
                        assert r['full_R4']
                        write(target/f'{label}_{output.request_id.replace("/","_")}.json',r)
                        results.append(dict(id=r['id'],policy=label,tokens=len(ids),finish_reason=reason,full_R4=True))
            print('QUALITY_COHORT',label,offset,len(group),flush=True)
        if control:
            control.close()
        e.close();del e,control,records
        gc.collect();torch.cuda.empty_cache()
    return dict(completed_quality_cases=results,scope='Full-answer accuracy pilot, not production capacity.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,required=True);args=parser.parse_args()
    register(args.budget)
