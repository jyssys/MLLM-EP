"""Small task-quality sanity before candidate evaluation. Not a GO benchmark."""
import os
import torch
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scripts.baseline import load,token_ids
from poc_async_recurrent_pp.emulator import generate_batch
from poc_async_recurrent_pp.scoring import score


@torch.inference_mode()
def main():
    model,tokenizer=load()
    data=read(ROOT/'outputs/manifests/data.json')
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    rows=sum([[r for r in data['pilot'] if r['dataset']==d][:4] for d in ['gsm8k','math500']],[])
    records=[]
    prompts=[token_ids(tokenizer,row,data['prompt_suffix']) for row in rows]
    results=generate_batch(model,tokenizer,prompts,8192)
    for row,ids,value in zip(rows,prompts,results):
        record=dict(id=row['id'],dataset=row['dataset'],prompt_ids=ids,**value,**score(row,value))
        write(output/(row['id'].replace('/','_')+'.json'),record)
        records.append(record)
        print('exact sanity',row['id'],record['correct_completed'],record['finish_reason'],len(record['ids']),flush=True)
    passing=all(sum(r['correct_completed'] for r in records if r['dataset']==d)>=1 for d in ['gsm8k','math500'])
    write(output/'status.json',dict(baseline_plausible=passing,n=8,correct=sum(r['correct_completed'] for r in records),
        note='Sanity only. It does not reproduce an official benchmark score.'))
    assert passing,'BASELINE_INVALID: task sanity requires investigation before approximation'


if __name__=='__main__':
    main()
