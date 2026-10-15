"""Problem-clustered scores. Capped generations are not correct completions."""
import csv
import argparse
import importlib.metadata
import statistics
import sys
from collections import defaultdict
import numpy as np
from poc_looped_exact_depth_chunk_pipeline.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer,aime_integer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'results/quality_native_v3')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ASSETS/'cpu_deps'))
    from math_verify import parse,verify,LatexExtractionConfig
    root=args.source;questions=read(ROOT/'results/frozen_quality_questions.json')
    gold={(r['dataset'],r['id']):r['answer'] for r in questions};records=[]
    for path in sorted(root.glob('*_cap*_wave*.json')):
        lane=path.name.split('_')[1]
        for row in read(path):
            grams=[tuple(row['generated_ids'][i:i+16]) for i in range(max(0,len(row['generated_ids'])-15))]
            answer,rule=final_answer(row['text']);expected=gold[(row['dataset'],row['problem_id'])]
            if row['dataset']=='aime24':
                normalized=aime_integer(answer);parsed=normalized is not None;correct=parsed and normalized==aime_integer(expected)
            else:
                normalized=answer;parsed=False;correct=False
                if answer is not None:
                    try:
                        a=parse('$'+expected+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                        b=parse('$'+answer+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                        parsed=bool(b);correct=bool(verify(a,b,strict=True,timeout_seconds=2))
                    except Exception:pass
            complete=row['finish_reason']=='stop'
            records.append({k:v for k,v in row.items() if k not in ['generated_ids','text','metrics']}|dict(
                lane=lane,extracted_answer=answer,normalized_answer=normalized,parse_rule=rule,parsed=parsed,
                reference=expected,answer_matches=bool(correct),complete=complete,correct_completed=bool(correct and complete),
                repeated_16gram_fraction=0. if not grams else 1-len(set(grams))/len(grams),
                request_latency_ms=(row['metrics']['last_token_ts']-row['metrics']['queued_ts'])*1000))
    if not records:return
    target=args.output/'quality_generations.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    grouped=defaultdict(list)
    for row in records:grouped[(row['dataset'],row['lane'],row['cap'])].append(row)
    summaries=[]
    for (dataset,lane,cap),rows in grouped.items():
        problems=defaultdict(list)
        for row in rows:problems[row['problem_id']].append(int(row['correct_completed']))
        means=np.array([np.mean(v) for v in problems.values()]);rng=np.random.default_rng(41015)
        samples=means[rng.integers(0,len(means),(5000,len(means)))].mean(1)
        expected=(90 if lane=='sampling' else 30) if dataset=='aime24' else 100
        by_seed={str(seed):dict(correct=sum(r['correct_completed'] for r in rows if r['seed']==seed),
            total=sum(r['seed']==seed for r in rows)) for seed in sorted({r['seed'] for r in rows})}
        summaries.append(dict(dataset=dataset,lane=lane,cap=cap,correct=sum(r['correct_completed'] for r in rows),
            total=len(rows),expected_total=expected,collection_complete=len(rows)==expected,
            score=float(means.mean()),problem_bootstrap_95CI=[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],
            independent_problems=len(problems),resamples=5000,by_seed=by_seed,
            natural_complete=sum(r['complete'] for r in rows),parsed=sum(r['parsed'] for r in rows),
            capped=sum(not r['complete'] for r in rows),mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in rows),
            median_generated_tokens=statistics.median(r['generated_tokens'] for r in rows),
            p95_generated_tokens=float(np.quantile([r['generated_tokens'] for r in rows],.95)),
            mean_repeated_16gram_fraction=statistics.mean(r['repeated_16gram_fraction'] for r in rows),
            capped_with_matching_extracted_answer=sum(r['answer_matches'] and not r['complete'] for r in rows),
            protocol='REFERENCE_PROTOCOL_NOT_IDENTICAL'))
    latest={}
    for row in summaries:
        key=row['dataset']+'_'+row['lane']
        if row['collection_complete'] and (key not in latest or row['cap']>latest[key]['cap']):latest[key]=row
    complete=len(latest)==3
    pass_completion=complete and all(r['natural_complete']/r['total']>=.9 for k,r in latest.items() if k.startswith('aime24'))
    write(args.output/'native_quality_r4.json',dict(all_caps=summaries,final_lanes=latest,source_directory=str(root.resolve()),
        quality_gate='QUALITY_BASELINE_PASS' if pass_completion else 'QUALITY_BASELINE_WEAK' if complete else 'COLLECTION_RUNNING',
        reference_protocol='REFERENCE_PROTOCOL_NOT_IDENTICAL',scorer=dict(math_verify=importlib.metadata.version('math-verify'),
        primary='Exact integer final answer in [0,999]. Capped outputs fail completed accuracy.',
        secondary='Math-Verify 0.8.0, LaTeX extraction only, strict equivalence, two-second timeouts.'),
        limitation='AIME has 30 independent problems. Seeds do not establish 1–2pp noninferiority or the private judge score.'))
    print([(r['dataset'],r['lane'],r['cap'],r['correct'],r['total']) for r in summaries])


if __name__=='__main__':main()
