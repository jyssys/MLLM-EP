"""Use pinned official LongBench metrics without selecting on answer quality."""
import collections
import importlib.util
import json
import sys
from .common import ROOT,MODEL,write,digest


def official():
    sys.path.insert(0,str(ROOT/'tools/quality_deps'))
    spec=importlib.util.spec_from_file_location('campaign_official_longbench_metrics',ROOT/'references/official_longbench_metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def scores(records,data):
    metrics=official();prompts={r['prompt_id']:r for r in data};values=[]
    tokenizer=None
    if any(row.get('text') is None for row in records):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    for row in records:
        prompt=prompts[row['prompt_id']];task=prompt['task'];text=row.get('text')
        if text is None:
            text=tokenizer.decode(row['output_ids'],skip_special_tokens=True)
        prediction=text.lstrip('\n').split('\n')[0] if task in ('triviaqa','samsum') else text
        metric=metrics.qa_f1_zh_score if task=='multifieldqa_zh' else metrics.rouge_zh_score if task=='dureader' else metrics.rouge_score if task in ('qmsum','samsum','multi_news','gov_report') else metrics.qa_f1_score
        score=max((metric(prediction,answer) for answer in prompt['reference_answers']),default=0.)
        values.append(dict(prompt_id=row['prompt_id'],task=task,score=100*score,
            empty_text=not text.strip(),metric=metric.__name__,output_tokens=len(row['output_ids'])))
    return values


def reference_audit():
    data=json.loads((ROOT/'data/candidate_pool_original.json').read_text())
    records=json.loads((ROOT/'data/natural_reference_all_candidates.json').read_text())['rows']
    values=scores(records,data);selected={r['prompt_id'] for r in json.loads((ROOT/'data/longbench_balanced400_v1.json').read_text())}
    summary={}
    for scope,rows in (('candidate_pool',values),('selected400',[r for r in values if r['prompt_id'] in selected])):
        summary[scope]={}
        for task in sorted({r['task'] for r in rows}):
            part=[r for r in rows if r['task']==task]
            summary[scope][task]=dict(n=len(part),metric=part[0]['metric'],mean_score=sum(r['score'] for r in part)/len(part),
                zero_score=sum(r['score']==0 for r in part),empty_text=sum(r['empty_text'] for r in part))
    result=dict(rows=values,summary=summary,official_metrics_SHA256=digest(ROOT/'references/official_longbench_metrics.py'),
        official_revision='2e00731f8d0bff23dc4325161044d0ed8af94c1e',
        evidence='TASK_QUALITY_AUDIT',
        scope='Official task metrics. QA scores and empty answers never change the frozen dataset. '
            'A nonzero benchmark score does not prove KV or layer correctness.')
    write(ROOT/'analysis/candidate_and_selected_task_quality.json',result)
    return summary


if __name__=='__main__':print(json.dumps(reference_audit(),ensure_ascii=False,indent=2))
