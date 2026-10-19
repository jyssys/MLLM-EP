"""Describe admitted original-mode outputs. This is not candidate validation."""
import argparse
import csv
import datetime as dt
import gzip
import json
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha


def compare_tokens(left,right):
    prefix=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)))
    equal=left==right
    return dict(output_token_ids_equal=equal,common_prefix_tokens=prefix,
                first_divergent_token=None if equal else prefix,
                fixed_output_tokens=len(left),adaptive_output_tokens=len(right))


def analyze(root,source):
    root,source=Path(root).resolve(),Path(source).resolve()
    summary=read(source)
    if summary['evidence_class']!='QUALITY_ONLY':
        raise ValueError('Use admitted baseline quality, not a hardware timing summary')
    by_key={(r['size'],r['mode'],r['id']):r for r in summary['records']}
    if len(by_key)!=len(summary['records']):
        raise ValueError('Admitted request identity repeats')
    hashes={str(source.relative_to(root)):sha(source)}
    pairs=[]
    sizes=sorted({row['size'] for row in summary['records']})
    for size in sizes:
        fixed={key[2] for key in by_key if key[:2]==(size,'fixed')}
        adaptive={key[2] for key in by_key if key[:2]==(size,'adaptive')}
        if not fixed or not adaptive:
            continue
        if fixed!=adaptive or len(fixed)!=64:
            raise ValueError('Paired output comparison requires both complete64-request arms')
        for key in sorted(fixed):
            rows=[by_key[(size,mode,key)] for mode in ('fixed','adaptive')]
            raw=[]
            for row in rows:
                path=root/row['source']
                expected=summary['input_sha256'][row['source']]
                if sha(path)!=expected:
                    raise ValueError('Admitted raw output changed')
                hashes[row['source']]=expected
                with gzip.open(path,'rt',encoding='utf-8') as stream:
                    raw.append(json.load(stream))
            if raw[0]['prompt_ids']!=raw[1]['prompt_ids']:
                raise ValueError('Paired inputs differ')
            pairs.append(dict(size=size,dataset=rows[0]['dataset'],id=key,
                **compare_tokens(raw[0]['output_ids'],raw[1]['output_ids']),
                fixed_correct=rows[0]['correct_completed'],adaptive_correct=rows[1]['correct_completed'],
                fixed_parsed=rows[0]['parsed'],adaptive_parsed=rows[1]['parsed'],
                fixed_finish=raw[0]['finish_reason'],adaptive_finish=raw[1]['finish_reason'],
                fixed_recorded_core_passes=rows[0]['recorded_decode_core_pass_work'],
                adaptive_recorded_core_passes=rows[1]['recorded_decode_core_pass_work'],
                fixed_source=rows[0]['source'],adaptive_source=rows[1]['source']))
    aggregates=[]
    for size in sizes:
        for domain in ('gsm8k','math500'):
            selected=[r for r in pairs if (r['size'],r['dataset'])==(size,domain)]
            if not selected:
                continue
            aggregates.append(dict(size=size,dataset=domain,N_pairs=len(selected),
                exact_output_matches=sum(r['output_token_ids_equal'] for r in selected),
                fixed_correct_adaptive_wrong=sum(r['fixed_correct'] and not r['adaptive_correct'] for r in selected),
                fixed_wrong_adaptive_correct=sum(not r['fixed_correct'] and r['adaptive_correct'] for r in selected),
                stop_reason_matches=sum(r['fixed_finish']==r['adaptive_finish'] for r in selected)))
    return dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),pairs=pairs,summaries=aggregates,
        input_sha256=hashes,evidence_class='QUALITY_ONLY',candidate_validation='NOT_RUN',
        rule='Original fixed/adaptive operating points may differ. No quality loss or speedup is attributed to a new mechanism.',
        limitations='No new logits or KV are captured. Different free-running trajectories do not support per-position causal work comparisons after divergence.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value=analyze(ROOT,args.source)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    for key in ('pairs','summaries'):
        rows=value[key]
        if rows:
            with (args.output/f'{key}.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(value['summaries']))


if __name__=='__main__':
    main()
