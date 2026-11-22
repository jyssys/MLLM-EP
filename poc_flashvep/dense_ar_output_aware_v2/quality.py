"""Use the official task-specific metric for every expanded LongBench task."""
import importlib.util
import sys
from .common import ROOT,PREVIOUS,MODEL,digest

def official():
    sys.path.insert(0,str(PREVIOUS/'tools/quality_deps'))
    path=PREVIOUS/'references/official_longbench_metrics.py'
    spec=importlib.util.spec_from_file_location('v2_pinned_official_longbench_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def scores(records,data):
    metrics=official();prompts={r['prompt_id']:r for r in data};values=[];tokenizer=None
    names={
        'multifieldqa_zh':'qa_f1_zh_score','dureader':'rouge_zh_score','vcsum':'rouge_zh_score',
        'qmsum':'rouge_score','samsum':'rouge_score','multi_news':'rouge_score','gov_report':'rouge_score',
        'trec':'classification_score','lsht':'classification_score','passage_count':'count_score',
        'passage_retrieval_en':'retrieval_score','passage_retrieval_zh':'retrieval_zh_score',
        'lcc':'code_sim_score','repobench-p':'code_sim_score'}
    for row in records:
        prompt=prompts[row['prompt_id']];task=prompt['task'];text=row.get('text')
        if text is None:
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
            text=tokenizer.decode(row['output_ids'],skip_special_tokens=True)
        prediction=text.lstrip('\n').split('\n')[0] if task in ('trec','triviaqa','samsum','lsht') else text
        metric=getattr(metrics,names.get(task,'qa_f1_score'))
        kwargs={'all_classes':prompt['all_classes']} if task in ('trec','lsht') else {}
        value=max((metric(prediction,answer,**kwargs) for answer in prompt['reference_answers']),default=0.)
        values.append(dict(prompt_id=row['prompt_id'],task=task,score=100*value,empty_text=not text.strip(),
            metric=metric.__name__,output_tokens=len(row['output_ids'])))
    return values
