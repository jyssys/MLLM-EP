"""Separate EOS handling, strict extraction, and nonterminal answer matches."""
import argparse
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cap',type=int,choices=[8192,16384,24576],required=True)
    args=parser.parse_args()
    root=ROOT/'results/quality_v1'
    assert read(root/f'aime24_sampling_cap{args.cap}_summary.json')['collection_complete']
    raw=[r for path in sorted(root.glob(f'aime24_sampling_cap{args.cap}_wave*.json')) for r in read(path)]
    scored=read(ROOT/f'results/aime_sampling_cap{args.cap}_complete_score.json')['records']
    assert len(raw)==len(scored)==90
    gold={r['id']:r['answer'] for r in read(ROOT/'results/frozen_questions.json') if r['dataset']=='aime24'}
    wrong=[dict(id=r['id'],seed=r['seed'],normalized_answer=r['normalized_answer'],
                reference=gold[r['id']],parse_rule=r['parse_rule'])
           for r in scored if r['finish_reason']=='stop' and not r['correct_completed']]
    result=dict(cap=args.cap,total=90,EOS_count=sum(r['finish_reason']=='stop' for r in raw),
        EOS_without_terminal_ID2=sum(r['finish_reason']=='stop' and r['generated_ids'][-1:]!=[2] for r in raw),
        EOS_before_last_position=sum(2 in r['generated_ids'][:-1] for r in raw),
        length_stops_containing_ID2=sum(r['finish_reason']=='length' and 2 in r['generated_ids'] for r in raw),
        backend_corruption_flags=sum(bool(r['metrics']['is_corrupted']) for r in raw),
        completed_unparsed=sum(r['finish_reason']=='stop' and r['normalized_answer'] is None for r in scored),
        completed_incorrect=wrong,
        capped_matching_extracted_answers=[dict(id=r['id'],seed=r['seed'],answer=r['normalized_answer'])
            for r in scored if r['finish_reason']=='length' and r['answer_matches']],
        grades_changed=False,stop_rules_changed=False,
        scope='Observed token and frozen-parser checks for one completed cap. Nonterminal matching answers do not count as completed accuracy.')
    target=ROOT/f'results/aime_cap{args.cap}_terminal_and_grade_audit.json'
    assert not target.exists()
    write(target,result)
    print({k:result[k] for k in ['EOS_count','EOS_without_terminal_ID2','EOS_before_last_position',
                                'length_stops_containing_ID2','completed_unparsed']})
    print('Completed incorrect:',wrong)
    print('Capped matching extracted answers:',len(result['capped_matching_extracted_answers']))


if __name__=='__main__':
    main()
