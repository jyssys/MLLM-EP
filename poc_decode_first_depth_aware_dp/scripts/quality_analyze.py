"""Paired request-level quality gate. Token loops are not statistical samples."""
import argparse
import csv
import statistics
import numpy as np
from poc_decode_first_depth_aware_dp.contract import *
from poc_async_recurrent_pp.scoring import score


def interval(values):
    a=np.asarray(values,dtype=float)
    draws=np.random.default_rng(20261018).integers(0,len(a),(20000,len(a)))
    return np.quantile(a[draws].mean(1),[.025,.975]).tolist()


def severe_pilot_stop(point,ci):
    return point>=15 and ci[0]>0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--cohort',choices=['pilot','full'],default='pilot');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/quality_protocol_v2.json')
    args=parser.parse_args();out=args.output;out.mkdir(exist_ok=False,parents=True)
    protocol=read(args.protocol)
    wanted={r['id']:r for r in protocol['rows'] if args.cohort=='full' or r['split']=='pilot'}
    observed={};scored=[]
    for folder in args.runs:
        assert read(folder/'status.json')['success']
        for path in sorted(folder.glob('*.json')):
            raw=read(path)
            if not isinstance(raw,dict) or 'depth_events' not in raw:continue
            assert raw['id'] in wanted
            key=raw['id'],raw['policy'];assert key not in observed
            result=score(wanted[raw['id']],dict(text=raw['text'],finish_reason='eos' if raw['finish_reason'] in ['eos','stop'] else 'length'))
            row=dict(id=raw['id'],dataset=raw['dataset'],policy=raw['policy'],
                answer_matches=result['answer_matches'],parsed=result['parsed'],extracted_answer=result['extracted_answer'],
                gold=wanted[raw['id']]['answer'],parser_version=result['parser_version'],
                finish_reason=raw['finish_reason'],truncated=raw['finish_reason']=='length',
                output_tokens=raw['output_length'],mean_depth=raw['mean_actual_decode_depth'],
                decode_loop_work=raw['total_decode_loop_work'],fixed_decode_loop_work=raw['full_depth_decode_work'],
                wall_ms=raw['wall_ms'],raw_path=str(path))
            observed[key]=row;scored.append(row)
    assert set(observed)=={(i,p) for i in wanted for p in ['fixed','dynamic']},'Do not reduce the registered denominator.'
    rows=[];failures=[]
    for dataset in ['gsm8k','math500']:
        ids=sorted(i for i,r in wanted.items() if r['dataset']==dataset)
        fixed=[observed[i,'fixed'] for i in ids];dynamic=[observed[i,'dynamic'] for i in ids]
        loss=[100*(int(f['answer_matches'])-int(d['answer_matches'])) for f,d in zip(fixed,dynamic)]
        ci=interval(loss);point=statistics.mean(loss)
        depth=[r['mean_depth'] for r in dynamic]
        entry=dict(dataset=dataset,n=len(ids),fixed_correct=sum(r['answer_matches'] for r in fixed),
            dynamic_correct=sum(r['answer_matches'] for r in dynamic),
            fixed_accuracy_percent=100*sum(r['answer_matches'] for r in fixed)/len(ids),
            dynamic_accuracy_percent=100*sum(r['answer_matches'] for r in dynamic)/len(ids),
            accuracy_drop_pp=point,paired_bootstrap_CI95_drop_pp=ci,
            paired_losses=sum(f['answer_matches'] and not d['answer_matches'] for f,d in zip(fixed,dynamic)),
            paired_gains=sum(d['answer_matches'] and not f['answer_matches'] for f,d in zip(fixed,dynamic)),
            fixed_parse_failures=sum(not r['parsed'] for r in fixed),dynamic_parse_failures=sum(not r['parsed'] for r in dynamic),
            fixed_truncations=sum(r['truncated'] for r in fixed),dynamic_truncations=sum(r['truncated'] for r in dynamic),
            mean_request_depth=statistics.mean(depth),request_bootstrap_CI95_depth=interval(depth),
            executed_decode_loops_saved_percent=100*(1-sum(r['decode_loop_work'] for r in dynamic)/sum(r['fixed_decode_loop_work'] for r in dynamic)),
            severe_pilot_failure=severe_pilot_stop(point,ci),
            evidence='MEASURED_ONLINE_FREE_RUNNING_GREEDY. Request-level paired bootstrap, not token-level inference.')
        rows.append(entry)
        for f,d in zip(fixed,dynamic):
            if f['answer_matches'] and not d['answer_matches']:failures.append(dict(id=f['id'],fixed=f,dynamic=d))
    severe=any(r['severe_pilot_failure'] for r in rows)
    verdict='QUALITY_FAIL' if severe else 'PILOT_UNCERTAIN_REQUIRES_FULL_QUALITY'
    if args.cohort=='full':
        allowed={'gsm8k':1.,'math500':2.}
        verdict=('ADAPTIVE_DECODE_VALID' if all(r['paired_bootstrap_CI95_drop_pp'][1]<=allowed[r['dataset']] for r in rows)
            else 'QUALITY_FAIL' if any(r['paired_bootstrap_CI95_drop_pp'][0]>allowed[r['dataset']] for r in rows) else 'INCONCLUSIVE')
    reference=read(ROOT/'artifacts/quality_pilot_preflight.json')['reference_status']
    if verdict=='ADAPTIVE_DECODE_VALID' and reference!='PASS':verdict='INCONCLUSIVE'
    write(out/'quality_summary.json',dict(rows=rows,verdict=verdict,cohort=args.cohort,
        bootstrap_repetitions=20000,sample_unit='Question ID, with both policies paired.',
        reference_qualification=reference,protocol_sha256=sha(args.protocol),
        no_threshold_rescue=True,RL=False,subsequent_stages_allowed=verdict=='ADAPTIVE_DECODE_VALID'))
    write(out/'quality_predictions.json',scored);write(out/'baseline_correct_candidate_wrong.json',failures)
    with (out/'quality_predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scored[0]),lineterminator='\n');writer.writeheader();writer.writerows(scored)
    print('QUALITY_GATE',verdict)
    for r in rows:print(r['dataset'],r['fixed_correct'],r['dynamic_correct'],r['n'],r['accuracy_drop_pp'],r['paired_bootstrap_CI95_drop_pp'])


if __name__=='__main__':main()
