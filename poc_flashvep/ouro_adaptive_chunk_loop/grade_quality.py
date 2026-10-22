"""Reuse the preserved evaluator. Keep accuracy and completion separate."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from .common import ROOT,write,digest

PRIOR=Path('/home/esjung/MLLM-EP-ouro-request-pp')

def evaluator():
    sys.path.append(str(PRIOR))
    path=PRIOR/'poc_async_recurrent_pp/scoring.py'
    spec=importlib.util.spec_from_file_location('frozen_ouro_quality_scoring',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score,dict(source=str(path),sha256=digest(path),
        extractor_sha256=digest(PRIOR/'poc_looped_exact_depth_chunk_pipeline/scoring.py'))

def main():
    scorer,identity=evaluator()
    manifest=json.loads((ROOT/'quality_manifest.json').read_text())
    cases={r['id']:r for r in manifest['rows']}
    rows=[]
    for folder in (ROOT/'raw').glob('p1_quality_*'):
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if not isinstance(data,dict) or data.get('id') not in cases or 'output_ids' not in data:
                continue
            case=cases[data['id']]
            complete=data['finish_reason'] in ('stop','frontend_registered_stop','eos')
            result=scorer(case,dict(text=data['text'],finish_reason='eos' if complete else 'length'))
            rows.append(dict(id=data['id'],dataset=data['dataset'],policy=data['policy'],
                correct=result['answer_matches'],parsed=result['parsed'],extracted=result['extracted_answer'],gold=case['answer'],
                complete=complete,correct_completed=result['correct_completed'],truncated=not complete,
                output_tokens=len(data['output_ids']),full_R4=data['full_R4'],source=str(file.relative_to(ROOT)),
                source_sha256=digest(file),evidence='LIVE_NATURAL_GENERATION_QUALITY_EXPLORATORY'))
    summaries=[]
    paired=[]
    rng=np.random.default_rng(20261023)
    for domain in ['gsm8k','math500']:
        for policy in sorted({r['policy'] for r in rows}):
            subset=[r for r in rows if r['dataset']==domain and r['policy']==policy]
            if not subset:
                continue
            summaries.append(dict(dataset=domain,policy=policy,N=len(subset),correct=sum(r['correct'] for r in subset),
                accuracy_percent=100*np.mean([r['correct'] for r in subset]),complete_correct=sum(r['correct_completed'] for r in subset),
                parsing_failures=sum(not r['parsed'] for r in subset),truncated=sum(r['truncated'] for r in subset),
                output_tokens=sum(r['output_tokens'] for r in subset)))
            if policy=='atomic':
                continue
            reference={r['id']:r for r in rows if r['dataset']==domain and r['policy']=='atomic'}
            matches=[r for r in subset if r['id'] in reference]
            if matches:
                delta=np.array([100*(int(r['correct'])-int(reference[r['id']]['correct'])) for r in matches])
                draws=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
                paired.append(dict(dataset=domain,candidate=policy,reference='atomic',paired_N=len(matches),delta_accuracy_pp=float(delta.mean()),
                    CI95_low=float(np.percentile(draws,2.5)),CI95_high=float(np.percentile(draws,97.5)),
                    limitation='Question-paired accuracy pilot. A wide interval does not prove equivalence.'))
    for name,values in [('quality_cases.csv',rows),('quality_summary.csv',summaries),('quality_paired_differences.csv',paired)]:
        if values:
            with (ROOT/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write(ROOT/'quality_evaluator_lock.json',identity)
    print(json.dumps(dict(cases=len(rows),summary=summaries,paired=paired),indent=2))

if __name__=='__main__':
    main()
