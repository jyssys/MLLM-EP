"""Score BI0 variation with the declared LongBench task metrics."""
import collections
import json
import re
import string
import sys
import numpy as np
from .common import ROOT,MODEL,write,digest

GOLD='BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1'


def qa_f1(prediction,answer):
    def tokens(text):
        text=''.join(c for c in text.lower() if c not in set(string.punctuation))
        text=re.sub(r'\b(a|an|the)\b',' ',text)
        return text.split()
    left,right=tokens(prediction),tokens(answer)
    common=sum((collections.Counter(left)&collections.Counter(right)).values())
    return 2*common/(len(left)+len(right)) if common else 0.


def scores(records,data):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    sys.path.insert(0,str(ROOT/'tools/quality_deps'))
    from rouge import Rouge
    rouge=Rouge()
    source={r['prompt_id']:r for r in data};values=[]
    for row in records:
        prompt=source[row['prompt_id']]
        text=tokenizer.decode(row['output_ids'],skip_special_tokens=True)
        options=[]
        for answer in prompt['reference_answers']:
            if prompt['task']=='qmsum':
                try:value=rouge.get_scores([text],[answer],avg=True)['rouge-l']['f']
                except (ValueError,RecursionError):value=0.
            else:value=qa_f1(text,answer)
            options.append(value)
        values.append(dict(prompt_id=row['prompt_id'],task=prompt['task'],
            score=100*max(options,default=0.),text=text,output_ids=row['output_ids']))
    return values


def quality_gate(name,margin=2.):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    actual=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    reference=json.loads((ROOT/'raw'/GOLD/'natural_reference.json').read_text())['rows']
    wanted={r['prompt_id'] for r in actual}
    reference=[r for r in reference if r['prompt_id'] in wanted]
    gold={r['prompt_id']:r for r in scores(reference,data)}
    measured=scores(actual,data)
    tasks={};rng=np.random.default_rng(2026112027)
    for task in sorted({r['task'] for r in measured}):
        part=[r for r in measured if r['task']==task]
        delta=np.array([r['score']-gold[r['prompt_id']]['score'] for r in part])
        draws=np.mean(rng.choice(delta,size=(10000,len(delta)),replace=True),axis=1)
        ci=[float(np.percentile(draws,p)) for p in (2.5,97.5)]
        tasks[task]=dict(n=len(part),reference_score=float(np.mean([gold[r['prompt_id']]['score'] for r in part])),
            actual_score=float(np.mean([r['score'] for r in part])),delta_mean=float(delta.mean()),
            paired_prompt_CI95=ci,point_noninferiority=bool(delta.mean()>=-margin),
            CI_noninferiority=ci[0]>=-margin,
            strict_token_matches=sum(r['output_ids']==gold[r['prompt_id']]['output_ids'] for r in part))
    lifetime={p.name:json.loads(p.read_text())['passed'] for p in folder.glob('KV_lifetime_*.json')}
    result=dict(name=name,n=len(actual),reference=GOLD,task_score_margin=margin,
        tasks=tasks,strict_tokens_matched=sum(r['output_ids']==gold[r['prompt_id']]['output_ids'] for r in measured),
        lifetime=lifetime,point_passed=len(actual)==cfg['count'] and bool(lifetime) and all(lifetime.values()) and all(v['point_noninferiority'] for v in tasks.values()),
        CI_passed=len(actual)==cfg['count'] and bool(lifetime) and all(lifetime.values()) and all(v['CI_noninferiority'] for v in tasks.values()),
        per_request=[dict(prompt_id=r['prompt_id'],task=r['task'],actual=r['score'],reference=gold[r['prompt_id']]['score'],
            strict_match=r['output_ids']==gold[r['prompt_id']]['output_ids']) for r in measured],
        evidence='BI0_TASK_QUALITY_EVALUATION',
        scope='Different geometry can change output. Prompt-paired quality bootstrap is distinct from server-start performance CI. Small-N task checks do not qualify full200 quality.')
    write(ROOT/'analysis'/(name+'_quality_gate.json'),result)
    return result


def reference_summary():
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    reference=json.loads((ROOT/'raw'/GOLD/'natural_reference.json').read_text())['rows']
    values=scores(reference,data)
    from .dataset import stats
    by_task={}
    for task in sorted({r['task'] for r in data}):
        part=[r for r in reference if r['task']==task]
        by_task[task]=dict(n=len(part),output=stats([r['output_tokens'] for r in part]),
            output_at_least100=sum(r['output_tokens']>=100 for r in part),cap_hits=sum(r['cap_hit'] for r in part),
            task_score=float(np.mean([r['score'] for r in values if r['task']==task])))
    result=dict(dataset_SHA256=digest(ROOT/'data/longbench_shortanswer200_v2.json'),
        reference_SHA256=digest(ROOT/'raw'/GOLD/'natural_reference.json'),
        input=stats([r['input_tokens'] for r in data]),output=stats([r['output_tokens'] for r in reference]),
        cap_hits=sum(r['cap_hit'] for r in reference),output_at_least100=sum(r['output_tokens']>=100 for r in reference),
        total_output_tokens=sum(r['output_tokens'] for r in reference),by_task=by_task,
        qasper_mean_below100=by_task['qasper']['output']['mean']<100,
        numerical_mode='BI0',evidence='MEASURED_NATURAL_OUTPUT_REFERENCE',
        scope='Natural EOS, commoncap2048. Length statistics do not measure online serving performance.')
    write(ROOT/'analysis/ShortAnswer200_v2_BI0_reference_summary.json',result)
    write(ROOT/'data/BI0_reference_task_scores.json',dict(rows=values))
    return result


if __name__=='__main__':
    print(json.dumps(reference_summary() if len(sys.argv)==1 else quality_gate(sys.argv[1]),indent=2))
