"""Full-answer quality lane. No forced output tokens or depth changes."""
import gc
import json
from pathlib import Path
import time
import torch
from .common import ROOT, MODEL, write, digest

SOURCE=Path('/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/dataset_manifest.json')

def freeze():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    source=json.loads(SOURCE.read_text())
    selected=[]
    for domain,count in [('gsm8k',64),('math500',50)]:
        for row in [r for r in source['rows'] if r['dataset']==domain][:count]:
            selected.append(dict(row,input_ids=tokenizer.encode(row['prompt'])))
    value=dict(rows=selected,source=str(SOURCE),source_sha256=digest(SOURCE),
        recipe='Preserved base-model few-shot GSM8K / Problem-Solution MATH recipe. Not the serving chat proxy.',
        max_new_tokens=4096,temperature=0,static_R=4,exit_threshold=1.0,
        stop_semantics='Native EOS or preserved per-case stop string. Caps remain truncated.',
        sample_selection='First64 GSM8K and first50 MATH by frozen source order. Input-only selection.')
    target=ROOT/'quality_manifest.json'
    assert not target.exists(),'Frozen quality inputs cannot change'
    write(target,value)
    jobs=[]
    policies=[['atomic',1024],['grouped',1024],['mixed',512]]
    for gpu in range(8):
        ids=[r['id'] for i,r in enumerate(selected) if i%8==gpu]
        order=policies[gpu%3:]+policies[:gpu%3]
        jobs.append(dict(id=f'p1_quality_pilot_gpu{gpu}_v1',kind='quality',gpu=gpu,cells=order,ids=ids,
            width=4,budget=2048,cap=4096,split='DIAGNOSTIC_QUALITY',timeout_s=14400))
    write(ROOT/'p1_quality_plan.json',dict(gpus=list(range(8)),jobs=jobs,
        prerequisite='Phase1 state/race review. Exploratory quality if numerical cause remains unresolved.'))

@torch.inference_mode()
def run_job(model,job,target):
    from transformers import AutoTokenizer
    from vllm_rlt import SamplingParams
    from .worker import engine,dispose
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    manifest=json.loads((ROOT/'quality_manifest.json').read_text())
    cases=[r for r in manifest['rows'] if r['id'] in job['ids']]
    summaries=[]
    for mode,chunk in job['cells']:
        e=engine(model,mode,chunk,job.get('width',4),job)
        # Warm full-R4 paths without forcing a benchmark answer.
        e.add_request('warm',tokenizer.encode('Q: What is two plus two?\nA:'),SamplingParams(max_tokens=8,max_loops=4,exit_threshold=1.,temperature=0))
        while e.has_unfinished_requests():
            e.step()
        for offset in range(0,len(cases),job.get('width',4)):
            group=cases[offset:offset+job.get('width',4)]
            pending=[]
            records={}
            for case in group:
                file=target/f"{mode}_{case['id'].replace('/','_')}.json"
                if file.exists():
                    continue
                rid=case['id']
                pending.append(case)
                records[rid]=dict(id=rid,dataset=case['dataset'],prompt=case['prompt'],prompt_ids=case['input_ids'],
                    gold=case['answer'],policy=mode,C=chunk,cap=job['cap'],output_ids=[],token_times=[],depths=[],
                    start=time.perf_counter(),evidence='LIVE_NATURAL_QUALITY_EXPLORATORY_UNTIL_BATCH_CAUSE_RESOLVED')
                e.add_request(rid,case['input_ids'],SamplingParams(max_tokens=job['cap'],max_loops=4,exit_threshold=1.,temperature=0))
            byid={r['id']:r for r in pending}
            while e.has_unfinished_requests():
                for out in e.step():
                    now=time.perf_counter()
                    r=records[out.request_id]
                    ids=list(out.token_ids)
                    if len(ids)>len(r['output_ids']):
                        assert len(ids)==len(r['output_ids'])+1
                        r['output_ids']=ids
                        r['depths']=list(out.exit_depths)
                        r['token_times'].append(now)
                    text=tokenizer.decode(ids,skip_special_tokens=False)
                    stop=next((s for s in byid[out.request_id]['stop_strings'] if s in text),None)
                    if out.finished or stop is not None:
                        reason=getattr(out.finish_reason,'value',out.finish_reason) if out.finished else 'frontend_registered_stop'
                        if not out.finished:
                            e.abort_request(out.request_id)
                        r.update(text=text,finish_reason=reason,registered_stop=stop,end=now,
                            output_tokens=len(ids),full_R4=all(d==4 for d in r['depths']),
                            E2E_s=now-r['start'],TTFT_s=r['token_times'][0]-r['start'],
                            TPOT_s=(now-r['token_times'][0])/(len(ids)-1) if len(ids)>1 else None)
                        write(target/f"{mode}_{out.request_id.replace('/','_')}.json",r)
                        summaries.append(dict(id=r['id'],policy=mode,tokens=len(ids),finish_reason=reason,full_R4=r['full_R4']))
            print('QUALITY_COHORT',mode,offset,len(pending),flush=True)
        dispose(e)
        del e
        gc.collect()
        torch.cuda.empty_cache()
    return dict(completed_quality_cases=summaries,
        scope='FULL_OUTPUT_FREE_RUNNING_GREEDY. Quality timing includes answer decoding. Not clean serving capacity.')

if __name__=='__main__':
    freeze()
