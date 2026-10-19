"""Score completed frozen calibration pilots. Never promote pilot accuracy to parity."""
import argparse
import collections
import csv
import datetime
import math
import random
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_async_recurrent_pp.scoring import score

def interval(values):
    rng=random.Random(2026101911);N=len(values)
    if not N:return None
    samples=sorted(statistics.mean(rng.choices(values,k=N)) for _ in range(4000))
    return [samples[100],samples[3899]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    protocol=read(ROOT/'artifacts/probe_protocol.json')
    by_id={r['id']:r for r in protocol['rows'] if r['campaign_partition']=='calibration'}
    records=[]
    for run in sorted((ROOT/'outputs').glob('quality_*')):
        for status in run.glob('*/status.json'):
            if not read(status).get('success'):continue
            for file in sorted(status.parent.glob('*.json')):
                value=read(file)
                if 'output_ids' not in value:continue
                row=by_id[value['id']]
                raw_finish_reason=value['finish_reason']
                if value.get('runtime')=='native_vllm' and value.get('native_finish_reason')=='stop' and value.get('native_stop_reason') is None:
                    value['finish_reason']='eos'
                evaluation=score(row,value)
                # A frozen stop string can complete a few-shot answer without
                # EOS. The source grader treats all non-EOS outcomes as truncation.
                # This campaign records native stop semantics explicitly.
                evaluation['truncated']=value['finish_reason']=='length'
                evaluation['correct_completed']=bool(evaluation['answer_matches'] and not evaluation['truncated'])
                records.append(dict(id=row['id'],dataset=row['dataset'],size=value['size'],mode=value['mode'],
                    runtime=value.get('runtime','staged_reference'),TP=value.get('TP',1),
                    generated_tokens=value['generated_tokens'],finish_reason=value['finish_reason'],
                    raw_finish_reason=raw_finish_reason,
                    wall_ms=value['wall_ms'],source=str(file.relative_to(ROOT)),**evaluation))
    assert records
    summaries=[]
    arms=sorted({(r['size'],r['mode'],r['runtime'],r['TP'],r['dataset']) for r in records})
    for size,mode,runtime,TP,domain in arms:
                rows=[r for r in records if (r['size'],r['mode'],r['runtime'],r['TP'],r['dataset'])==(size,mode,runtime,TP,domain)]
                ids=[r['id'] for r in rows];assert len(ids)==len(set(ids)),(size,mode,runtime,TP,domain)
                bits=[int(r['correct_completed']) for r in rows]
                summaries.append(dict(size=size,mode=mode,runtime=runtime,TP=TP,dataset=domain,N=len(rows),correct=sum(bits),
                    accuracy_percent=100*statistics.mean(bits),accuracy_CI_percent=[100*x for x in interval(bits)],
                    parsing_failures=sum(not r['parsed'] for r in rows),truncated=sum(r['truncated'] for r in rows),
                    EOS=sum(r['finish_reason']=='eos' for r in rows),
                    frozen_stop=sum(r['finish_reason']=='stop' for r in rows),
                    median_output_tokens=statistics.median(r['generated_tokens'] for r in rows)))
    comparisons=[]
    for size in MODELS:
        for domain in ['gsm8k','math500']:
            pair={mode:{r['id']:r for r in records if (r['size'],r['mode'],r['dataset'],r['runtime'])==(size,mode,domain,'staged_reference')}
                  for mode in ['fixed','adaptive']}
            ids=sorted(set(pair['fixed'])&set(pair['adaptive']))
            if not ids:continue
            diff=[int(pair['adaptive'][i]['correct_completed'])-int(pair['fixed'][i]['correct_completed']) for i in ids]
            comparisons.append(dict(size=size,dataset=domain,N_pairs=len(ids),
                adaptive_minus_fixed_pp=100*statistics.mean(diff),
                paired_bootstrap_95_CI_pp=[100*x for x in interval(diff)],
                caution='Operating-point context only. This is not a new mechanism quality change or a noninferiority proof.'))
    backend=[]
    for size in MODELS:
        for domain in ['gsm8k','math500']:
            reference={r['id']:r for r in records if (r['size'],r['mode'],r['runtime'],r['dataset'])==(size,'fixed','staged_reference',domain)}
            for TP in [1,2,4]:
                native={r['id']:r for r in records if (r['size'],r['mode'],r['runtime'],r['TP'],r['dataset'])==(size,'fixed','native_vllm',TP,domain)}
                ids=sorted(set(reference)&set(native))
                if not ids:continue
                diff=[int(native[i]['correct_completed'])-int(reference[i]['correct_completed']) for i in ids]
                token_checks=[]
                for key in ids:
                    left=read(ROOT/reference[key]['source'])['output_ids']
                    right=read(ROOT/native[key]['source'])['output_ids']
                    first=next((j for j,(a,b) in enumerate(zip(left,right)) if a!=b),None)
                    if first is None and len(left)!=len(right):first=min(len(left),len(right))
                    noEOS_left=left[:-1] if left and left[-1]==protocol['EOS'] else left
                    noEOS_right=right[:-1] if right and right[-1]==protocol['EOS'] else right
                    token_checks.append(dict(id=key,raw_equal=left==right,first_raw_divergence=first,
                        terminal_EOS_ignored_equal=noEOS_left==noEOS_right,
                        reference_count=len(left),native_count=len(right)))
                backend.append(dict(size=size,dataset=domain,TP=TP,N_pairs=len(ids),
                    native_minus_staged_accuracy_pp=100*statistics.mean(diff),
                    paired_bootstrap_95_CI_pp=[100*x for x in interval(diff)],token_checks=token_checks,
                    caution='Baseline backend comparison. No candidate transformation is active. Small calibration N does not prove quality equivalence.'))
    write(args.output,dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evidence_class='QUALITY_ONLY',records=records,summaries=summaries,comparisons=comparisons,
        native_vs_staged_baseline=backend,
        parser_source='poc_async_recurrent_pp/scoring.py',parser_sha256=sha(PROJECT/'poc_async_recurrent_pp/scoring.py'),
        completion_rule='Only native length cap is truncation. EOS or frozen stop counts as completed.',
        native_EOS_rule='Pinned vLLM v1/core/sched/utils.py:104-106 marks EOS without assigning stop_reason. Analysis derives EOS from native_finish_reason=stop and native_stop_reason=None. Raw files remain unchanged.',
        cohort='Frozen calibration, 8/domain/arm. Held-out items remain untouched.',
        scope='Baseline pilot only. No candidate accuracy, performance, or equivalence claim.'))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
