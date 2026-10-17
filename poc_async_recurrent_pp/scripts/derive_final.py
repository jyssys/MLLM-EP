"""Rescore native generations without changing raw outputs or their original grades."""
import argparse
import csv
import importlib.metadata
import numpy as np
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.scoring import score,PARSER_VERSION
from poc_async_recurrent_pp.scripts.aggregate import summarize,collect


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wave',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/final_analysis')
    args=parser.parse_args()
    output=args.output.resolve()
    assert output.is_relative_to(ROOT/'outputs') and not output.exists()
    output.mkdir()
    gold={row['id']:row for row in read(ROOT/'outputs/manifests/data.json')['full']}
    original={arm:{row['id']:row for row in collect(args.wave,arm)} for arm in ['exact','stale','delta','w0']}
    ids=set(original['exact'])
    assert all(set(rows)==ids for rows in original.values())
    canonical={}
    audit=[]
    for arm,rows in original.items():
        destination=output/'canonical'/arm
        canonical[arm]={}
        for identifier,row in rows.items():
            assert row['native_fixed_batch'] is True, 'Rejected custom-driver results cannot enter final analysis'
            assert row['prompt_ids']==original['exact'][identifier]['prompt_ids']
            assert row['ids'][0]==original['exact'][identifier]['ids'][0], 'Exact-prefill first token mismatch'
            updated=score(gold[identifier],row)
            record={key:row[key] for key in ['id','dataset','generated_tokens','finish_reason']}
            record.update(updated)
            record['source']=str((args.wave/arm/(identifier.replace('/','_')+'.json')).resolve().relative_to(ROOT))
            record['original_parser_version']=row.get('parser_version','parpp-final-v1')
            canonical[arm][identifier]=record
            write(destination/(identifier.replace('/','_')+'.json'),record)
            audit.append(dict(id=identifier,arm=arm,original_parsed=row['parsed'],canonical_parsed=updated['parsed'],
                original_answer_correct=row['answer_matches'],canonical_answer_correct=updated['answer_matches'],
                original_completed_correct=row['correct_completed'],canonical_completed_correct=updated['correct_completed'],
                original_extracted=row['extracted_answer'],canonical_extracted=updated['extracted_answer']))
        write(destination/'status.json',dict(success=True,n=len(rows),source='Independent CPU rescore, not new model generation'))
    rows=summarize(output/'canonical')
    write(output/'canonical/paired_summary.json',dict(rows=rows,bootstrap_resamples=10000,sample_unit='Prompt'))
    transitions=[]
    examples=[]
    for arm in ['stale','delta','w0']:
        for identifier in sorted(ids):
            base=original['exact'][identifier]
            candidate=original[arm][identifier]
            a=canonical['exact'][identifier]
            b=canonical[arm][identifier]
            shortest=min(len(base['ids']),len(candidate['ids']))
            first=next((i for i in range(shortest) if base['ids'][i]!=candidate['ids'][i]),None)
            if first is None and len(base['ids'])!=len(candidate['ids']):
                first=shortest
            transition=dict(id=identifier,dataset=b['dataset'],arm=arm,
                baseline_correct=a['answer_matches'],candidate_correct=b['answer_matches'],
                baseline_EOS=not a['truncated'],candidate_EOS=not b['truncated'],
                baseline_tokens=base['generated_tokens'],candidate_tokens=candidate['generated_tokens'],
                first_divergent_token=first,
                positional_token_agreement=float(np.mean(np.array(base['ids'][:shortest])==np.array(candidate['ids'][:shortest]))),
                scope='Free-running positional comparison. Prefixes after divergence differ.')
            transitions.append(transition)
            if a['answer_matches'] and not b['answer_matches']:
                examples.append(transition|dict(reference_answer=gold[identifier]['answer'],
                    baseline_answer=a['extracted_answer'],candidate_answer=b['extracted_answer'],
                    candidate_parsed=b['parsed'],baseline_source=a['source'],candidate_source=b['source']))
    rejected={arm:[row['dataset'] for row in rows if row['arm']==arm
                  and row['answer_delta_pp_CI95'][1]<-10 and row['delta_pp_CI95'][1]<-10]
              for arm in ['stale','delta','w0']}
    clear_failure=all(rejected.values())
    decision=dict(verdict='QUALITY_FAIL' if clear_failure else 'INCONCLUSIVE',
        basis='A candidate must pass both required benchmarks. A large paired regression on either mandatory benchmark rejects that candidate.',
        negative_rule='Each candidate has CI upper < -10 pp on both answer and completed-answer accuracy in at least one mandatory dataset.',
        rejected_on_benchmarks=rejected,full_GSM8K_1319='NOT_RUN',full_MATH500_500='NOT_RUN',
        distributed_latency_oracle='NOT_RUN',distributed_prototype='NOT_RUN',dynamic_depth='NOT_RUN',
        source_wave=str(args.wave.resolve().relative_to(ROOT)),parser_version=PARSER_VERSION,
        scorer_sha256=sha(ROOT/'scoring.py'),predictor_sha256=sha(ROOT/'outputs/manifests/predictor.json'),
        independent_prompts=len(ids),summary=rows,
        raw_scoring_changes=sum(row['original_answer_correct']!=row['canonical_answer_correct'] for row in audit),
        completed_scoring_changes=sum(row['original_completed_correct']!=row['canonical_completed_correct'] for row in audit),
        limitation='Pilot rejection only. It does not estimate complete benchmark accuracy or demonstrate architecture speedup.')
    write(output/'final_verdict.json',decision)
    write(output/'scorer_audit.json',audit)
    write(output/'correctness_transitions.json',transitions)
    write(output/'baseline_right_candidate_wrong.json',examples)
    with (output/'accuracy_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    records=[record for data in canonical.values() for record in data.values()]
    with (output/'request_scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if args.wave.name=='pilot_cap8192_v2':
        earlier=ROOT/'outputs/pilot_cap2048_v2'
        prefix=[]
        for arm,data in original.items():
            for identifier,current in data.items():
                old=read(earlier/arm/(identifier.replace('/','_')+'.json'))
                common=min(len(old['ids']),len(current['ids']))
                prefix.append(dict(id=identifier,arm=arm,common_tokens=common,
                    common_prefix_equal=old['ids'][:common]==current['ids'][:common],
                    note='Different caps and earlier client history can change allocation history. Do not assume pure censoring.'))
        write(output/'cross_cap_prefix_audit.json',prefix)
    print(decision['verdict'],decision['rejected_on_benchmarks'],flush=True)
    print([(r['dataset'],r['arm'],r['answer_correct'],r['n'],r['answer_delta_pp'],r['answer_delta_pp_CI95']) for r in rows])


if __name__=='__main__':
    main()
