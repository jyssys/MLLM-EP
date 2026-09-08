"""Distributional quality analysis; descriptive strata are not causal claims."""
import argparse,ast,json,re
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
df=pd.read_csv(a.input);a.out.mkdir(parents=True,exist_ok=True)
def answer_type(raw):
    answers=ast.literal_eval(raw)
    values=[str(x).strip().lower() for x in answers]
    if all(v in ('yes','no') for v in values):return 'yes_no'
    if all(re.fullmatch(r'[+\-]?[\d,.]+%?',v) for v in values):return 'numeric'
    return 'text'
df['answer_type']=df.answers.map(answer_type)
base=df[df.policy=='vanilla'][['request_id','correct','prediction','output_tokens']]
pairs=df.merge(base,on='request_id',suffixes=('','_baseline'),validate='many_to_one')
pairs['quality_delta_pp']=100*(pairs.correct-pairs.correct_baseline)
pairs['output_length']=pairs.output_tokens.map(lambda x:len(ast.literal_eval(x)))
pairs['baseline_output_length']=pairs.output_tokens_baseline.map(lambda x:len(ast.literal_eval(x)))
pairs['assignment_change_fraction']=pairs.changed_or_skipped_assignments/pairs.counted_assignments
summary=[]
for (dataset,policy,kind),g in pairs.groupby(['dataset','policy','answer_type']):
    summary.append(dict(dataset=dataset,policy=policy,answer_type=kind,n=len(g),
        baseline_accuracy_percent=100*g.correct_baseline.mean(),accuracy_percent=100*g.correct.mean(),
        delta_pp=g.quality_delta_pp.mean(),lost_correct=int(((g.correct_baseline==1)&(g.correct==0)).sum()),
        gained_correct=int(((g.correct_baseline==0)&(g.correct==1)).sum()),
        median_assignment_change_fraction=g.assignment_change_fraction.median(),
        output_length_mean=g.output_length.mean(),baseline_output_length_mean=g.baseline_output_length.mean(),
        interpretation='DESCRIPTIVE_STRATUM_NOT_CAUSAL; thresholds not fitted to this held-out set'))
pd.DataFrame(summary).to_csv(a.out/'failure_strata.csv',index=False)
pairs[(pairs.correct_baseline==1)&(pairs.correct==0)].to_csv(a.out/'lost_correct_examples.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
