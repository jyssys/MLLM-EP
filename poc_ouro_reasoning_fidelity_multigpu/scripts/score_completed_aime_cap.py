"""Transparent AIME scoring for one complete cap, not a final quality decision."""
import argparse
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_ouro_reasoning_fidelity_multigpu.scripts.analyze_quality import problem_ci
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer,aime_integer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cap',type=int,choices=[8192,16384,24576],required=True)
    args=parser.parse_args()
    source=ROOT/'results/quality_v1'
    assert read(source/f'aime24_sampling_cap{args.cap}_summary.json')['collection_complete']
    rows=[r for path in sorted(source.glob(f'aime24_sampling_cap{args.cap}_wave*.json')) for r in read(path)]
    assert len(rows)==90
    assert len({(r['problem_id'],r['seed']) for r in rows})==90
    gold={r['id']:aime_integer(r['answer']) for r in read(ROOT/'results/frozen_questions.json') if r['dataset']=='aime24'}
    assert {r['problem_id'] for r in rows}==set(gold)
    assert all({r['seed'] for r in rows if r['problem_id']==i}==set(SEEDS) for i in gold)
    scored=[]
    for row in rows:
        answer,rule=final_answer(row['text'])
        normalized=aime_integer(answer)
        matches=normalized is not None and normalized==gold[row['problem_id']]
        scored.append(dict(id=row['problem_id'],seed=row['seed'],finish_reason=row['finish_reason'],
            generated_tokens=row['generated_tokens'],extracted_answer=answer,parse_rule=rule,
            normalized_answer=normalized,answer_matches=matches,
            correct_completed=matches and row['finish_reason']=='stop'))
    target=ROOT/f'results/aime_sampling_cap{args.cap}_complete_score.json'
    assert not target.exists()
    write(target,dict(cap=args.cap,correct_completed=sum(r['correct_completed'] for r in scored),total=90,
        EOS=sum(r['finish_reason']=='stop' for r in scored),capped=sum(r['finish_reason']=='length' for r in scored),
        problem_cluster_accuracy=problem_ci([(r['id'],int(r['correct_completed'])) for r in scored]),
        by_seed={str(seed):dict(correct=sum(r['correct_completed'] for r in scored if r['seed']==seed),
            total=sum(r['seed']==seed for r in scored)) for seed in SEEDS},
        records=scored,protocol='REFERENCE_PROTOCOL_NOT_IDENTICAL',
        decision='No final readiness decision. Continue every frozen whole-lane extension required by protocol_lock.json.',
        prompt_choice_unchanged=True,final_native_score_file='results/quality_score/native_quality_r4.json'))
    print(dict(cap=args.cap,correct=sum(r['correct_completed'] for r in scored),total=90,
               EOS=sum(r['finish_reason']=='stop' for r in scored)))


if __name__=='__main__':
    main()
