"""Score admitted native outputs. Keep backend differences separate from methods."""
import argparse
import datetime as dt
import gzip
import json
from pathlib import Path
import statistics
from poc_ouro_parallelism_deep_discovery.contract import ROOT,read,sha
from poc_ouro_parallelism_deep_discovery.scripts.native_quality_extension import require
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval
from poc_ouro_parallelism_deep_discovery.scripts.baseline_output_comparison import compare_tokens
from poc_async_recurrent_pp.scoring import score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--integrity',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Use a fresh analysis directory')
    integrity=read(args.integrity)
    for relative,digest in integrity['input_sha256'].items():
        require(sha(ROOT/relative)==digest,'Admitted source changed: '+relative)
    protocol=read(ROOT/'artifacts/baseline_extension_protocol_v1.json')
    frozen={r['id']:r for r in protocol['rows']}
    staged=read(ROOT/'analysis/baseline_extension_complete_v1/summary.json')
    reference={(r['size'],r['id']):r for r in staged['records'] if r['mode']=='fixed'}
    records=[]
    for job in integrity['jobs']:
        if job['status']!='PASS':
            continue
        require(len(job['records'])==64,'Native integrity arm is incomplete')
        for item in job['records']:
            with gzip.open(ROOT/item['source'],'rt',encoding='utf-8') as stream:
                value=json.load(stream)
            evaluated=score(frozen[value['id']],value)
            evaluated['truncated']=value['finish_reason']=='length'
            evaluated['correct_completed']=bool(evaluated['answer_matches'] and not evaluated['truncated'])
            previous=reference[(job['size'],value['id'])]
            require(sha(ROOT/previous['source'])==staged['input_sha256'][previous['source']],'Staged raw changed')
            with gzip.open(ROOT/previous['source'],'rt',encoding='utf-8') as stream:
                old=json.load(stream)
            require(old['prompt_ids']==value['prompt_ids'],'Backend comparison has different input IDs')
            parity=compare_tokens(old['output_ids'],value['output_ids'])
            parity['staged_output_tokens']=parity.pop('fixed_output_tokens')
            parity['native_output_tokens']=parity.pop('adaptive_output_tokens')
            records.append(dict(size=job['size'],id=value['id'],dataset=value['dataset'],source=item['source'],
                generated_tokens=len(value['output_ids']),finish_reason=value['finish_reason'],
                staged_correct=previous['correct_completed'],**evaluated,
                **parity))
    summaries=[]
    for size,domain in sorted({(r['size'],r['dataset']) for r in records}):
        selected=[r for r in records if (r['size'],r['dataset'])==(size,domain)]
        require(len(selected)==len({r['id'] for r in selected})==32,'Native domain is incomplete')
        bits=[int(r['correct_completed']) for r in selected]
        differences=[int(r['correct_completed'])-int(r['staged_correct']) for r in selected]
        summaries.append(dict(size=size,dataset=domain,N=32,correct=sum(bits),accuracy_percent=100*statistics.mean(bits),
            accuracy_CI_percent=[100*x for x in interval(bits)],truncated=sum(r['truncated'] for r in selected),
            parsing_failures=sum(not r['parsed'] for r in selected),raw_output_matches=sum(r['output_token_ids_equal'] for r in selected),
            native_minus_staged_pp=100*statistics.mean(differences),paired_CI_pp=[100*x for x in interval(differences)]))
    result=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),records=records,summaries=summaries,
        input_sha256={str(args.integrity.resolve().relative_to(ROOT)):sha(args.integrity),
            'analysis/baseline_extension_complete_v1/summary.json':sha(ROOT/'analysis/baseline_extension_complete_v1/summary.json')},
        evidence_class='QUALITY_ONLY',candidate_quality='NOT_RUN',candidate_E2E='NOT_RUN',
        limitation='Same intended fixedR4 function, different BF16 backend and native string versus staged token-suffix stopping. No numerical-equivalence or architecture-speedup claim.')
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(summaries))


if __name__=='__main__':
    main()
