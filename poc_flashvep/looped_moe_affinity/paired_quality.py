"""Paired downstream and held-out NLL comparisons from actual saved scores."""
import csv
from collections import Counter
import json
import numpy as np
from .common import ROOT
from .aggregate_quality import csv_write

def ifm_records(model,ep):
    pattern=(f'{model}_official_*/official/eval_results/**/*.jsonl' if ep==1 else
        f'{model}_ep{ep}_official_quality_v*/official.rank0/eval_results/**/*.jsonl')
    result={};occurrences=Counter()
    for path in (ROOT/'raw').glob(pattern):
        # An active evaluator can append a record. Consume only LF-complete records.
        for line in path.read_bytes().split(b'\n')[:-1]:
            if not line:continue
            record=json.loads(line)
            task='mmlu' if 'mmlu' in path.parts else 'arc_challenge'
            if task=='arc_challenge' and 'arc_challenge' not in path.name:continue
            base=(task,path.stem,record['resume_hashes']['item'])
            ordinal=occurrences[base];occurrences[base]+=1
            key=(*base,ordinal)
            assert key not in result
            field='acc_token' if task=='arc_challenge' else 'acc'
            result[key]=dict(correct=record['metrics'][field]/100,prompt_hash=record['resume_hashes']['prompt'])
    return result

def main():
    rows=[];samples=[];rng=np.random.default_rng(20261026)
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        native=ifm_records(model,1) if model!='foil1' else {}
        for ep in [4,8]:
            actual=ifm_records(model,ep) if model!='foil1' else {}
            if model=='foil1':
                for task in ['lambada_standard','lambada_openai','xwinograd_en']:
                    nfile=ROOT/'raw'/f'foil1_official_{task}_full_v1/foil_results.rank0.json'
                    efile=ROOT/'raw'/f'foil1_ep{ep}_official_quality_v1/foil_results.rank0.json'
                    if not efile.exists():continue
                    for target,path in [(native,nfile),(actual,efile)]:
                        result=json.loads(path.read_text())
                        for value in result['samples'].get(task,[]):
                            target[(task,value['doc_hash'])]=dict(correct=float(value['acc']),prompt_hash=value['prompt_hash'])
            for task in sorted(set(k[0] for k in actual)):
                keys=[k for k in actual if k[0]==task and k in native]
                differences=[]
                for key in keys:
                    n,a=native[key],actual[key]
                    assert n['prompt_hash']==a['prompt_hash'],('PROMPT_MISMATCH',model,ep,key)
                    delta=a['correct']-n['correct'];differences.append(delta)
                    samples.append(dict(model=model,ep=ep,benchmark=task,item_key=str(key[1:]),native_correct=n['correct'],ep_correct=a['correct']))
                n=len(differences)
                if not n:continue
                differences=np.asarray(differences)
                # Use bounded chunks for full MMLU. Do not allocate a 4,000 x 14,042 matrix.
                bootstrap=np.concatenate([rng.choice(differences,(100,n),replace=True).mean(1) for _ in range(40)])
                rows.append(dict(model=model,ep=ep,benchmark=task,paired_n=n,delta_pp=100*float(differences.mean()),
                    ci95_low_pp=100*float(np.quantile(bootstrap,.025)),ci95_high_pp=100*float(np.quantile(bootstrap,.975)),
                    native_correct_to_ep_wrong=int(np.sum(differences<0)),native_wrong_to_ep_correct=int(np.sum(differences>0)),
                    completion='FULL_BENCHMARK' if n==dict(mmlu=14042,arc_challenge=1165,lambada_standard=5153,lambada_openai=5153,xwinograd_en=2325)[task] else 'PARTIAL_BENCHMARK',
                    evidence='LIVE_PAIRED_COMPLETED_SUBTASK_RESULTS'))
    csv_write(ROOT/'paired_accuracy.csv',rows)
    csv_write(ROOT/'quality/paired_item_scores.csv',samples)
    nll=[]
    native=ROOT/'raw/foil1_ep1_official_fineweb_nll_v1/nll_sequences.rank0.jsonl'
    if native.exists():
        expected={v['sequence']:v for v in map(json.loads,native.read_text().splitlines())}
        for ep in [1,4,8]:
            p=ROOT/'raw'/f'foil1_ep{ep}_official_fineweb_nll_v1/nll_sequences.rank0.jsonl'
            if not p.exists():continue
            actual=[json.loads(line) for line in p.read_text().splitlines()]
            values=np.array([v['nll'] for v in actual]);delta=values-np.array([expected[v['sequence']]['nll'] for v in actual])
            boot=np.concatenate([rng.choice(delta,(100,len(delta)),replace=True).mean(1) for _ in range(40)])
            nll.append(dict(model='foil1',ep=ep,sequences=len(values),tokens=sum(v['tokens'] for v in actual),nll=float(values.mean()),
                native_delta_nats=float(delta.mean()),delta_ci95_low=float(np.quantile(boot,.025)),delta_ci95_high=float(np.quantile(boot,.975)),
                reference_nll=2.4492,evidence='LIVE_SHIFTED_TOKEN_LM_NLL',
                qualification='Paired sequence bootstrap. Adjacent windows can share source documents. Corpus recipe matches, paper binary hash unpublished.'))
    csv_write(ROOT/'quality/heldout_nll.csv',nll)

if __name__=='__main__':main()
