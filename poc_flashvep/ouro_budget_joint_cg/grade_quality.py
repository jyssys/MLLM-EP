"""Preserved evaluator, matched questions, and paired accuracy uncertainty."""
import json
import sys
import numpy as np
from .common import ROOT,write,digest
from .cost_predictor import csv_file
from ..ouro_adaptive_chunk_loop.grade_quality import evaluator

def main():
    sys.path.insert(0,'/home/esjung/MLLM-EP-ouro-request-pp')
    scorer,identity=evaluator()
    extractor=sys.modules['poc_looped_exact_depth_chunk_pipeline.scoring'].__file__
    assert digest(extractor)==identity['extractor_sha256'],'Actual imported evaluator differs from its frozen source'
    identity['actual_imported_extractor']=extractor
    manifest=json.loads((ROOT/'quality_manifest.json').read_text())
    cases={r['id']:r for r in manifest['rows']};rows=[];tokens={}
    for folder in (ROOT/'raw').glob('cb_p6_quality_*'):
        if not (folder/'completed.json').exists():
            continue
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if not isinstance(data,dict) or data.get('id') not in cases or 'output_ids' not in data:
                continue
            complete=data['finish_reason'] in ('stop','frontend_registered_stop','eos')
            score=scorer(cases[data['id']],dict(text=data['text'],finish_reason='eos' if complete else 'length'))
            rows.append(dict(id=data['id'],dataset=data['dataset'],policy=data['policy'],correct=score['answer_matches'],
                correct_complete=score['correct_completed'],parsed=score['parsed'],extracted=score['extracted_answer'],
                gold=cases[data['id']]['answer'],complete=complete,truncated=not complete,
                output_tokens=len(data['output_ids']),full_R4=data['full_R4'],file=str(file.relative_to(ROOT)),sha256=digest(file)))
            tokens[(data['policy'],data['id'])]=data['output_ids']
    summary=[];paired=[];rng=np.random.default_rng(20261023)
    for domain in ('gsm8k','math500'):
        reference={r['id']:r for r in rows if r['dataset']==domain and r['policy']=='F0A'}
        for policy in sorted({r['policy'] for r in rows}):
            values=[r for r in rows if r['dataset']==domain and r['policy']==policy]
            if not values:
                continue
            summary.append(dict(dataset=domain,policy=policy,N=len(values),correct=sum(r['correct'] for r in values),
                accuracy_pct=100*np.mean([r['correct'] for r in values]),correct_complete=sum(r['correct_complete'] for r in values),
                parse_failures=sum(not r['parsed'] for r in values),truncated=sum(r['truncated'] for r in values),
                output_tokens=sum(r['output_tokens'] for r in values)))
            if reference and policy!='F0A':
                matches=[r for r in values if r['id'] in reference]
                delta=np.array([100*(int(r['correct'])-int(reference[r['id']]['correct'])) for r in matches])
                draws=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
                paired.append(dict(dataset=domain,candidate=policy,reference='F0A',N=len(matches),
                    delta_accuracy_pp=float(delta.mean()),CI95_low=float(np.percentile(draws,2.5)),CI95_high=float(np.percentile(draws,97.5)),
                    token_exact=sum(tokens[(policy,r['id'])]==tokens[('F0A',r['id'])] for r in matches),
                    scope='Question-paired public quality pilot. Wide intervals do not prove equivalence.'))
    csv_file(ROOT/'quality_cases.csv',rows);csv_file(ROOT/'quality_summary.csv',summary);csv_file(ROOT/'quality_paired_differences.csv',paired)
    write(ROOT/'quality_evaluator_lock.json',identity)
    print(json.dumps(dict(cases=len(rows),summary=summary,paired=paired),indent=2))

if __name__=='__main__':
    main()
