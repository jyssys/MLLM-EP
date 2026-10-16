"""Grade bounded reference completions without replacing benchmark outputs."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer, aime_integer


def main():
    source=ROOT/'results/hf_prompt_corroboration_v1'
    assert read(source/'guard_status.json')['success']
    raw=read(source/'reference_completions.json')
    assert len(raw)==4
    assert {(r['problem_id'],r['arm']) for r in raw}=={(i,a) for i in ['60','61'] for a in ['P0','P1']}
    gold={r['id']:r['answer'] for r in read(ROOT/'results/frozen_questions.json') if r['dataset']=='aime24'}
    records=[]
    for row in raw:
        answer,rule=final_answer(row['text'])
        matches=aime_integer(answer) is not None and aime_integer(answer)==aime_integer(gold[row['problem_id']])
        expected=len(row['prompt_ids'])+row['generated_tokens']-1
        assert len(row['cache_lengths'])==192
        records.append(dict(id=row['problem_id'],arm=row['arm'],seed=row['seed'],
            generated_tokens=row['generated_tokens'],finish_reason=row['finish_reason'],
            extracted_answer=answer,parse_rule=rule,reference=gold[row['problem_id']],
            answer_matches=matches,correct_completed=matches and row['finish_reason']=='stop',
            terminal_EOS=row['generated_ids'][-1:]==[2],
            cache_slots=row['cache_slots'],expected_cache_length=expected,
            all_cache_lengths_match=all(n==expected for n in row['cache_lengths'])))
    selector=read(source/'hf_threshold_audit.json')
    target=ROOT/'results/HF_corroboration_summary.json'
    assert not target.exists()
    write(target,dict(records=records,independent_problems=2,
        official_reference_environment='Transformers 4.57.1 and SDPA, not the broken 4.54.1 combination.',
        default_selector_calibration=dict(teacher_steps=selector['teacher_steps'],
            actual_forward_exit_at_step=selector['actual_forward_exit_at_step'],
            hypothetical_choice_counts_R1_R2_R3_R4=selector['default_choice_counts_R1_R2_R3_R4'],
            snapshots=selector['snapshots'],alternative_quality='NOT_RUN'),
        whole_benchmark_scores_changed=False,prompt_choice_changed=False,
        scope='Four bounded reference completions on two previously declared overlapping problems. Not HF AIME30 accuracy.',
        timing_scope='Reference completion wall is not optimized native serving latency.'))


if __name__=='__main__':
    main()
