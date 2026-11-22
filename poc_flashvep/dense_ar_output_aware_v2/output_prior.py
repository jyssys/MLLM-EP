"""Freeze observable output priors without reading heldout EOS into features."""
import argparse
import collections
import hashlib
import json
import math
import time
import numpy as np
from .common import ROOT,write,digest
from .c0_analysis import csv_save

def output_bin(value):
    return 0 if value<=10 else 1 if value<=50 else 2 if value<=100 else 3 if value<=200 else 4

def reference():
    state=json.loads((ROOT/'state/official_compiled_c1_reference_complete_v2.json').read_text());assert state['passed']
    return list(map(json.loads,(ROOT/'raw'/state['name']/'requests.jsonl').read_text().splitlines()))

def freeze():
    path=ROOT/'configs/output_prior_models_frozen_v2.json';assert not path.exists()
    train=json.loads((ROOT/'data/balanced400_v2_discovery.json').read_text());ids={r['prompt_id'] for r in train}
    # Filter identities before extracting output labels. Heldout EOS never enters fit.
    labels={r['prompt_id']:r['output_tokens'] for r in reference() if r['prompt_id'] in ids};assert len(labels)==len(train)
    global_log=float(np.mean([math.log1p(labels[r['prompt_id']]) for r in train]));pseudo=8
    bytask=collections.defaultdict(list);byinput=collections.defaultdict(list);joint=collections.defaultdict(list)
    for row in train:
        value=math.log1p(labels[row['prompt_id']]);bytask[row['task']].append(value)
        byinput[str(row['input_bucket'])].append(value);joint[row['task']+':'+str(row['input_bucket'])].append(value)
    task={k:(sum(v)+pseudo*global_log)/(len(v)+pseudo) for k,v in bytask.items()}
    input_only={k:(sum(v)+pseudo*global_log)/(len(v)+pseudo) for k,v in byinput.items()}
    both={k:(sum(v)+pseudo*task[k.rsplit(':',1)[0]])/(len(v)+pseudo) for k,v in joint.items()}
    result=dict(time=time.time(),global_log=global_log,task=task,input_only=input_only,task_input=both,pseudocount=pseudo,
        training_prompt_IDs=sorted(ids),training_dataset_SHA256=digest(ROOT/'data/balanced400_v2_discovery.json'),
        online_features=['application-provided task label','tokenized input bucket'],output_cap=2048,
        future_EOS_feature=False,heldout_labels_used_for_fit=False,
        models=['Constant','InputOnly','TaskPrior','TaskPlusInput'],
        definition='Predict log1p output length with eight pseudo-observations. Shrink task/input means toward the training global mean. '
            'Shrink a task×input cell toward its task prior. Unseen tasks use the global prior. Clip predictions to[1,2048].',
        scope='The task is observable application metadata,not inferred after completion. The common max_tokens2048 is known but non-discriminative. '
            'Natural output remains unknown online. This selected balanced benchmark is not a production-distribution claim. '
            'Output forecasting alone does not validate an axis selector,safe window or serving speedup.')
    write(path,result)
    heldout=json.loads((ROOT/'data/balanced400_v2_heldout.json').read_text());predictions=[]
    assert not ids.intersection(r['prompt_id'] for r in heldout)
    for row in heldout:
        values=predict(result,row)
        predictions.append(dict(prompt_id=row['prompt_id'],task=row['task'],input_tokens=row['input_tokens'],input_bucket=row['input_bucket'],
            context_SHA256=row['context_SHA256'],predictions=values))
    write(ROOT/'data/heldout_output_prior_predictions_frozen_v2.json',dict(time=time.time(),rows=predictions,
        model_SHA256=digest(path),input_dataset_SHA256=digest(ROOT/'data/balanced400_v2_heldout.json'),
        future_EOS_feature=False,actual_output_labels_in_this_file=False))
    print('OBSERVABLE_OUTPUT_PRIORS_AND_HELDOUT_PREDICTIONS_FROZEN',len(train),len(heldout),flush=True)

def predict(model,row):
    task=model['task'].get(row['task'],model['global_log']);bucket=str(row['input_bucket'])
    values=dict(Constant=model['global_log'],InputOnly=model['input_only'].get(bucket,model['global_log']),
        TaskPrior=task,TaskPlusInput=model['task_input'].get(row['task']+':'+bucket,task))
    return {key:max(1.,min(2048.,math.expm1(value))) for key,value in values.items()}

def evaluate():
    frozen=ROOT/'data/heldout_output_prior_predictions_frozen_v2.json';record=json.loads(frozen.read_text())
    assert not record['future_EOS_feature'] and not record['actual_output_labels_in_this_file']
    labels={r['prompt_id']:r['output_tokens'] for r in reference()};rows=[];models={}
    for row in record['rows']:
        actual=labels[row['prompt_id']]
        for name,value in row['predictions'].items():
            rows.append(dict(prompt_id=row['prompt_id'],task=row['task'],input_bucket=row['input_bucket'],context_SHA256=row['context_SHA256'],
                model=name,predicted_tokens=value,actual_reference_tokens=actual,absolute_error_tokens=abs(actual-value),
                absolute_log_error=abs(math.log1p(actual)-math.log1p(value)),correct_output_bin=output_bin(actual)==output_bin(value),
                actual_output_bin=output_bin(actual),predicted_output_bin=output_bin(value)))
    for name in record['rows'][0]['predictions']:
        part=[r for r in rows if r['model']==name]
        models[name]=dict(n=len(part),mean_absolute_error_tokens=float(np.mean([r['absolute_error_tokens'] for r in part])),
            median_absolute_error_tokens=float(np.median([r['absolute_error_tokens'] for r in part])),
            mean_absolute_log_error=float(np.mean([r['absolute_log_error'] for r in part])),
            output_bin_accuracy=float(np.mean([r['correct_output_bin'] for r in part])),
            overflow_reference_requests=sum(r['actual_output_bin']==4 for r in part))
    result=dict(time=time.time(),prediction_file_SHA256=digest(frozen),models=models,
        evidence='HELDOUT_OUTPUT_PREDICTION',axis_selection_serving_gain='NOT_MEASURED',
        scope='Evaluate the frozen192 heldout predictions against the independent original compiled C1 natural reference. '
            'Reference labels are targets,not online features. Overflow>200 is retained as a fifth output category. '
            'Request-level prediction metrics are not independent-server-start performance evidence. '
            'A C0 cost lookup is not a validated compiled C3 queue model. No dynamic serving or oracle headroom follows from prediction accuracy.')
    write(ROOT/'analysis/heldout_output_prior_prediction_v2.json',result)
    csv_save(ROOT/'figures/heldout_output_prior_prediction_rows_v2.csv',rows)
    csv_save(ROOT/'figures/heldout_output_prior_prediction_summary_v2.csv',[dict(model=k,**v) for k,v in models.items()])
    print('HELDOUT_OUTPUT_PRIOR',models,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','evaluate'));args=p.parse_args();globals()[args.action]()
