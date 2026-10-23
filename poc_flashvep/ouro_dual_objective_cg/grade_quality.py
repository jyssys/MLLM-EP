"""Frozen evaluator and question-level paired intervals for new quality arms."""
import json
import sys
import numpy as np
from .common import ROOT,write,digest
from ..ouro_adaptive_chunk_loop.grade_quality import evaluator
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    sys.path.insert(0,'/home/esjung/MLLM-EP-ouro-request-pp')
    scorer,identity=evaluator()
    actual=sys.modules['poc_looped_exact_depth_chunk_pipeline.scoring'].__file__
    assert digest(actual)==identity['extractor_sha256']
    identity['actual_imported_extractor']=actual
    manifest=json.loads((ROOT/'quality_manifest.json').read_text());cases={r['id']:r for r in manifest['rows']}
    rows=[];outputs={};losses=[];prompt_ledgers=[]
    for folder in sorted((ROOT/'raw').glob('dual_p4_QUALITY_*')):
        if not (folder/'completed.json').exists():continue
        for ledger_file in folder.glob('*_prefill_work.json'):
            ledger=json.loads(ledger_file.read_text())
            prompt_ledgers.append(dict(run_id=folder.name,policy=ledger['policy'],
                full_R4_prompt=ledger['full_R4_prompt'],source=str(ledger_file.relative_to(ROOT)),sha256=digest(ledger_file)))
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if not isinstance(data,dict) or data.get('id') not in cases or 'output_ids' not in data:continue
            complete=str(data['finish_reason']).split('.')[-1].lower() in ('stop','eos','frontend_registered_stop')
            score=scorer(cases[data['id']],dict(text=data['text'],finish_reason='eos' if complete else 'length'))
            rows.append(dict(id=data['id'],dataset=data['dataset'],policy=data['policy'],correct=score['answer_matches'],
                correct_complete=score['correct_completed'],parsed=score['parsed'],extracted=score['extracted_answer'],
                gold=cases[data['id']]['answer'],complete=complete,truncated=not complete,
                finish_reason=data['finish_reason'],
                output_tokens=len(data['output_ids']),full_R4=data['full_R4'],file=str(file.relative_to(ROOT)),sha256=digest(file)))
            outputs[(data['policy'],data['id'])]=data['output_ids']
    summaries=[];pairs=[];direct=[];rng=np.random.default_rng(2026102303)
    for domain in ('gsm8k','math500'):
        reference={r['id']:r for r in rows if r['policy']=='F0A' and r['dataset']==domain}
        for policy in sorted({r['policy'] for r in rows}):
            values=[r for r in rows if r['dataset']==domain and r['policy']==policy]
            if not values:continue
            summaries.append(dict(dataset=domain,policy=policy,N=len(values),correct=sum(r['correct'] for r in values),
                accuracy_pct=100*np.mean([r['correct'] for r in values]),correct_complete=sum(r['correct_complete'] for r in values),
                complete_accuracy_pct=100*np.mean([r['correct_complete'] for r in values]),
                parse_failures=sum(not r['parsed'] for r in values),truncated=sum(r['truncated'] for r in values),
                output_tokens=sum(r['output_tokens'] for r in values)))
            if policy=='F0A':continue
            matched=[r for r in values if r['id'] in reference]
            if not matched:continue
            d=np.array([100*(int(r['correct'])-int(reference[r['id']]['correct'])) for r in matched])
            samples=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(axis=1)
            pairs.append(dict(dataset=domain,candidate=policy,reference='F0A',N=len(d),delta_accuracy_pp=float(d.mean()),
                CI95_low=float(np.percentile(samples,2.5)),CI95_high=float(np.percentile(samples,97.5)),
                token_exact=sum(outputs[(policy,r['id'])]==outputs[('F0A',r['id'])] for r in matched),
                scope='Question-paired historical public pilot. Zero observed difference is not a population equivalence proof.'))
            for r in matched:
                if reference[r['id']]['correct'] and not r['correct']:
                    losses.append(dict(dataset=domain,policy=policy,id=r['id'],candidate_file=r['file'],
                        baseline_file=reference[r['id']]['file'],gold=r['gold'],candidate_answer=r['extracted']))
        joint={r['id']:r for r in rows if r['policy']=='O6_J' and r['dataset']==domain}
        for control in ('O6_G','O6_G1024','O6_C','O6_CA','BEST_FIXED','MIXED_STOCK','MIXED_FAIR'):
            ref={r['id']:r for r in rows if r['policy']==control and r['dataset']==domain}
            ids=sorted(set(joint)&set(ref))
            if not ids:continue
            d=np.array([100*(int(joint[k]['correct'])-int(ref[k]['correct'])) for k in ids])
            dc=np.array([100*(int(joint[k]['correct_complete'])-int(ref[k]['correct_complete'])) for k in ids])
            indices=rng.integers(0,len(d),size=(5000,len(d)))
            boot=d[indices].mean(axis=1);complete_boot=dc[indices].mean(axis=1)
            direct.append(dict(dataset=domain,candidate='O6_J',reference=control,N=len(ids),
                delta_accuracy_pp=float(d.mean()),CI95_low=float(np.percentile(boot,2.5)),CI95_high=float(np.percentile(boot,97.5)),
                delta_complete_accuracy_pp=float(dc.mean()),complete_CI95_low=float(np.percentile(complete_boot,2.5)),
                complete_CI95_high=float(np.percentile(complete_boot,97.5)),
                token_exact=sum(outputs[('O6_J',k)]==outputs[(control,k)] for k in ids),
                output_count_equal=sum(len(outputs[('O6_J',k)])==len(outputs[(control,k)]) for k in ids),
                stop_reason_equal=sum(joint[k]['finish_reason']==ref[k]['finish_reason'] for k in ids),
                control_correct_to_wrong=sum(ref[k]['correct'] and not joint[k]['correct'] for k in ids),
                control_wrong_to_correct=sum(not ref[k]['correct'] and joint[k]['correct'] for k in ids),
                scope='Measured question-paired public pilot. Small-sample exploratory intervals, not population noninferiority proof.'))
    csv_file(ROOT/'quality_cases.csv',rows);csv_file(ROOT/'quality_summary.csv',summaries)
    csv_file(ROOT/'quality_paired_comparisons.csv',pairs);csv_file(ROOT/'quality_baseline_correct_to_wrong.csv',losses)
    csv_file(ROOT/'quality_matched_controls.csv',direct);csv_file(ROOT/'quality_prefill_ledger_validation.csv',prompt_ledgers)
    write(ROOT/'quality_evaluator_lock.json',identity)
    write(ROOT/'quality_analysis_status.json',dict(generations=len(rows),all_reported_R4=all(r['full_R4'] for r in rows),
        prefill_ledgers=len(prompt_ledgers),all_prefill_ledgers_R4=bool(prompt_ledgers) and all(r['full_R4_prompt'] for r in prompt_ledgers),
        scope='Measured frozen public pilot. Native submission accounting plus exit-depth assertion, not a universal numerical-equivalence proof.'))
    print(json.dumps(dict(generations=len(rows),summary=summaries,paired=pairs),indent=2))

if __name__=='__main__':main()
