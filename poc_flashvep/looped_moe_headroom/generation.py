"""Free-running rollout after actual expert interventions. Not a timing benchmark."""
import json
import re
import sys
import time
import torch

from .common import ROOT, XLLM, append, write
from .probes import Controller, oracle_plan
from .study import arms, rows

def answer(text,domain):
    boxed=[]
    start=text.rfind('\\boxed{')
    if start>=0:
        depth=1
        for end in range(start+7,len(text)):
            depth+=int(text[end]=='{')-int(text[end]=='}')
            if depth==0:
                boxed=[text[start+7:end]]
                break
    marked=re.findall(r'####\s*([^\n]+)',text)
    final=re.findall(r'(?:final answer|answer is|answer:)\s*([^\n]+)',text,flags=re.I)
    if boxed:
        return boxed[-1].strip(), 'boxed'
    if marked:
        return marked[-1].strip(), 'gsm_marker'
    if final:
        return final[-1].strip().rstrip('.'), 'explicit_answer'
    return None,'PARSING_FAILURE'

def normalize(text):
    if text is None:
        return None
    return str(text).replace(',','').replace('$','').replace(' ','').strip().rstrip('.')

@torch.inference_mode()
def rollout(a,ids,spec,cap,stop_strings=()):
    generated=[]
    work_total=work_removed=0
    cache_peak=0
    reason='length'
    start=time.perf_counter()
    for step in range(cap):
        if spec is None:
            logits=a.forward(ids)
        else:
            config={k:v for k,v in spec.items() if k!='id'}
            plan=None
            if config['mode'].startswith('oracle'):
                # Recompute an exact diagnostic baseline for the current candidate prefix.
                # This offline procedure is not deployable and has no speedup claim.
                with Controller(a,'probe') as probe:
                    a.forward(ids)
                    plan=oracle_plan(probe.frames)
            with Controller(a,**config,oracle=plan) as controller:
                logits=a.forward(ids)
                work_total+=controller.work['total_edges']
                work_removed+=controller.work['omitted_edges']
                cache_peak=max(cache_peak,controller.work['cache_peak_tensor_bytes'])
        token=logits[:,-1].argmax(-1,keepdim=True)
        generated.append(token.item())
        ids=torch.cat([ids,token],dim=-1)
        if generated[-1]==a.eos():
            reason='eos'
            break
        text=a.decode(generated)
        if any(stop in text for stop in stop_strings):
            reason='registered_stop'
            break
    return dict(ids=generated,text=a.decode(generated),finish_reason=reason,truncated=reason=='length',
        measured_wall_s=time.perf_counter()-start,
        total_expert_edges=work_total,omitted_expert_edges=work_removed,
        cache_peak_tensor_bytes=cache_peak,
        evidence='LIVE_FREE_RUNNING_GREEDY_QUALITY_NOT_CLEAN_TIMING')

def run(a,target,job):
    selected=rows(job)
    by_id={s['id']:s for s in arms()}
    options=[None]+[by_id[i] for i in job.get('arms',['O1_drop_0.1_renorm0','C1_router_0.2','C2_reuse_guardNone'])]
    n=0
    for row in selected:
        text=row['text']
        style=job.get('prompt_style','frozen_raw_text')
        if style=='official_ifm_gsm8k_8shot' and row['dataset']=='gsm8k':
            sys.path.insert(0,str(XLLM))
            from xllm.paper_part1.tasks.gsm8k import GSM8KTask
            task=object.__new__(GSM8KTask)
            shots=task.get_fewshot({},None)
            text=task.sep.join([task.textify_example(v) for v in shots]+[task.get_prompt(row)])
        elif style=='official_ifm_gsm8k_8shot':
            style='zero_shot_Problem_Answer_not_official_math_fewshot'
        uncut=a.encode(text)
        ids=uncut[:,:job.get('prefix_cap',512)]
        baseline=None
        for spec in options:
            result=rollout(a,ids,spec,job.get('cap',16),job.get('stop_strings',()))
            arm='B0' if spec is None else spec['id']
            if spec is None:
                baseline=result
            parsed,parser=answer(result['text'],row['dataset'])
            exact=result['ids']==baseline['ids']
            first=next((i for i,(x,y) in enumerate(zip(result['ids'],baseline['ids'])) if x!=y),None)
            append(target/'generation_quality.jsonl',dict(model=a.name,id=row['id'],dataset=row['dataset'],
                split=row['split'],arm=arm,prompt_ids=ids[0].tolist(),gold=row.get('gold'),cap=job.get('cap',16),
                prompt_style=style,input_truncated=uncut.shape[-1]!=ids.shape[-1],
                parsed_answer=parsed,parser=parser,correct=parsed is not None and normalize(parsed)==normalize(row.get('gold')),
                token_exact=exact,first_divergent_token=first,
                quality_scope='CAPPED_PILOT_NOT_DOWNSTREAM_ACCURACY' if job.get('cap',16)<256 else 'UNDERPOWERED_DOWNSTREAM_PILOT',
                control_budget='Nominal q only during free-running generation. Not exact matched work after output divergence.',
                **result))
            n+=1
        print('GENERATION_SEQUENCE',a.name,row['id'],flush=True)
    return dict(status='PASS',model=a.name,generated_sequences=n,
        quality_scope='No quality equivalence claim from short or truncated output')
