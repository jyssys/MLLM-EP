"""Paired prompt-bootstrap statistics. Tokens and loops are not samples."""
import argparse
import csv
import numpy as np
from poc_async_recurrent_pp.contract import *


def collect(wave, arm):
    assert read(wave/arm/'status.json')['success'], 'Require a successful complete worker'
    return [read(path) for path in sorted((wave/arm).glob('*.json'))
            if path.name not in ['status.json','availability_trace.json','eval_contract.json']]


def summarize(wave):
    values={arm:collect(wave,arm) for arm in ['exact','stale','delta','w0']}
    results=[]
    for dataset in ['gsm8k','math500']:
        baseline={row['id']:row for row in values['exact'] if row['dataset']==dataset}
        for arm,rows in values.items():
            selected={row['id']:row for row in rows if row['dataset']==dataset}
            assert set(selected)==set(baseline), 'Require complete paired collection'
            ids=sorted(baseline)
            if not ids:
                continue
            x=np.array([int(selected[i]['correct_completed']) for i in ids])
            a=np.array([int(baseline[i]['correct_completed']) for i in ids])
            answer=np.array([int(selected[i]['answer_matches']) for i in ids])
            answer_a=np.array([int(baseline[i]['answer_matches']) for i in ids])
            rng=np.random.default_rng(20261017)
            sample=rng.integers(0,len(ids),(10000,len(ids)))
            ci=np.quantile((x-a)[sample].mean(1)*100,[.025,.975]).tolist()
            results.append(dict(dataset=dataset,arm=arm,n=len(ids),correct=int(x.sum()),
                generation_errors=0,EOS_completed=sum(not selected[i]['truncated'] for i in ids),
                accuracy_percent=float(x.mean()*100),delta_pp=float((x-a).mean()*100),
                answer_correct=int(answer.sum()),answer_accuracy_percent=float(answer.mean()*100),
                answer_delta_pp=float((answer-answer_a).mean()*100),
                answer_delta_pp_CI95=np.quantile((answer-answer_a)[sample].mean(1)*100,[.025,.975]).tolist(),
                answer_accuracy_CI95=np.quantile(answer[sample].mean(1)*100,[.025,.975]).tolist(),
                delta_pp_CI95=ci,accuracy_CI95=np.quantile(x[sample].mean(1)*100,[.025,.975]).tolist(),
                baseline_right_candidate_wrong=int(((a==1)&(x==0)).sum()),
                baseline_wrong_candidate_right=int(((a==0)&(x==1)).sum()),
                parsed=sum(selected[i]['parsed'] for i in ids),
                truncated=sum(selected[i]['truncated'] for i in ids),
                capped_answer_matches=sum(selected[i]['answer_matches'] and selected[i]['truncated'] for i in ids),
                mean_tokens=float(np.mean([selected[i]['generated_tokens'] for i in ids])),
                median_tokens=float(np.median([selected[i]['generated_tokens'] for i in ids])),
                max_tokens=max(selected[i]['generated_tokens'] for i in ids),
                answer_baseline_error_ids=[i for i in ids if baseline[i]['answer_matches'] and not selected[i]['answer_matches']],
                baseline_error_ids=[i for i in ids if baseline[i]['correct_completed'] and not selected[i]['correct_completed']]))
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wave',type=Path,required=True)
    args=parser.parse_args()
    rows=summarize(args.wave)
    write(args.wave/'paired_summary.json',dict(rows=rows,bootstrap_resamples=10000,
        seed=20261017,sample_unit='Prompt',inference='Pilot screening only. Full datasets are required for GO.'))
    with (args.wave/'paired_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print([(r['dataset'],r['arm'],r['correct'],r['n'],r['delta_pp'],r['delta_pp_CI95']) for r in rows])


if __name__=='__main__':
    main()
