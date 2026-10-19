"""Compare natural outputs from existing native scheduler options."""
import argparse
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scored',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--left-runtime',default='native_vllm')
    parser.add_argument('--right-runtime',default='native_vllm_async')
    args=parser.parse_args()
    assert not args.output.exists()
    content=read(args.scored)
    records=content['records']
    pairs=[]
    for size in MODELS:
        for TP in [1,2,4]:
            for domain in ['gsm8k','math500']:
                left={r['id']:r for r in records if (r['size'],r['TP'],r['runtime'],r['dataset'])==(size,TP,args.left_runtime,domain)}
                right={r['id']:r for r in records if (r['size'],r['TP'],r['runtime'],r['dataset'])==(size,TP,args.right_runtime,domain)}
                ids=sorted(set(left)&set(right))
                if not ids:continue
                differences=[]
                details=[]
                for identifier in ids:
                    a,b=read(ROOT/left[identifier]['source']),read(ROOT/right[identifier]['source'])
                    assert a['prompt_ids']==b['prompt_ids']
                    assert a['revision']==b['revision'] and a['cap']==b['cap']
                    difference=int(right[identifier]['correct_completed'])-int(left[identifier]['correct_completed'])
                    differences.append(difference)
                    x,y=a['output_ids'],b['output_ids']
                    first=next((i for i,(v,w) in enumerate(zip(x,y)) if v!=w),None)
                    if first is None and len(x)!=len(y):first=min(len(x),len(y))
                    details.append(dict(id=identifier,raw_token_ids_equal=x==y,
                        first_divergence=first,default_tokens=len(x),async_tokens=len(y),
                        default_finish=left[identifier]['finish_reason'],async_finish=right[identifier]['finish_reason'],
                        default_correct=left[identifier]['correct_completed'],async_correct=right[identifier]['correct_completed'],
                        default_source=left[identifier]['source'],async_source=right[identifier]['source']))
                pairs.append(dict(size=size,TP=TP,dataset=domain,N=len(ids),
                    default_correct=sum(left[i]['correct_completed'] for i in ids),
                    async_correct=sum(right[i]['correct_completed'] for i in ids),
                    accuracy_delta_pp=100*statistics.mean(differences),
                    paired_request_bootstrap_95_CI_pp=[100*x for x in interval(differences)],
                    raw_token_parity=sum(d['raw_token_ids_equal'] for d in details),details=details))
    write(args.output,dict(pairs=pairs,evidence_class='QUALITY_ONLY',
        source=str(args.scored),source_sha256=sha(args.scored),
        left_runtime=args.left_runtime,right_runtime=args.right_runtime,
        scope='Existing native scheduler option. Frozen calibration only. Small N does not establish noninferiority.',
        timing='Natural latency differs when output work changes. This artifact makes no scheduling speedup claim.',
        candidate='NONE'))
    print('QUALITY_OPTION_PAIRS',sum(r['N'] for r in pairs))

if __name__=='__main__':main()
