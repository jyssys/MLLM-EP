"""Compare actual free-running outputs without inferring numerical equivalence."""
import collections
from .contract import *
from .serving_analysis import table


def first_difference(a,b):
    for index,(left,right) in enumerate(zip(a,b)):
        if left!=right:return index
    return min(len(a),len(b)) if len(a)!=len(b) else None


def main():
    rows=[json.loads(line) for line in (ROOT/'evaluation_outputs.jsonl').read_text().splitlines()]
    groups=collections.defaultdict(dict)
    for row in rows:
        if row.get('prompt_ID_parity',True):groups[row['run']][row['id']]=row
    comparisons=[('quality400_'+arm,'quality400_SD' if arm.endswith('D') else 'quality400_S4')
        for arm in ARMS if arm not in ['S4','SD']]
    comparisons+=[('quality400_SD','quality400_S4'),
        ('quality400_native_TP4','quality400_native_TP1'),
        ('quality400_native_TP1','quality400_S4'),
        ('CDB_SD_pilot_v1','quality_SD_pilot_v1')]
    details=[];summaries=[]
    for candidate,reference in comparisons:
        common=sorted(groups[candidate].keys() & groups[reference].keys())
        for rid in common:
            x,y=groups[candidate][rid],groups[reference][rid]
            a,b=read(ROOT/x['source']),read(ROOT/y['source'])
            assert a['prompt_ids']==b['prompt_ids'],(candidate,reference,rid)
            token_difference=first_difference(a['output_ids'],b['output_ids'])
            depth_difference=first_difference(a['depths'],b['depths']) if 'depths' in a and 'depths' in b else None
            details.append(dict(candidate=candidate,reference=reference,id=rid,dataset=x['dataset'],
                token_exact=token_difference is None,first_different_generated_index=token_difference,
                reference_output_tokens=len(b['output_ids']),candidate_output_tokens=len(a['output_ids']),
                depth_comparable='depths' in a and 'depths' in b,
                depth_exact=depth_difference is None if 'depths' in a and 'depths' in b else None,
                first_different_depth_index=depth_difference,
                reference_decode_loop_work=y['decode_loop_work'],candidate_decode_loop_work=x['decode_loop_work'],
                reference_correct=y['correct'],candidate_correct=x['correct'],
                reference_stop=b['finish_reason'],candidate_stop=a['finish_reason'],
                text_exact=a['text']==b['text'],reference_source=y['source'],candidate_source=x['source'],
                evidence='LIVE_MEASURED free-running generation. First differences do not identify a unique numerical cause.'))
        for domain in ['gsm8k','math500']:
            selected=[r for r in details if r['candidate']==candidate and r['reference']==reference and r['dataset']==domain]
            if not selected:continue
            summaries.append(dict(candidate=candidate,reference=reference,dataset=domain,N_common=len(selected),
                token_exact=sum(r['token_exact'] for r in selected),
                depth_comparable=sum(r['depth_comparable'] for r in selected),
                depth_exact=sum(r['depth_exact'] is True for r in selected),
                same_tokens_different_rendering=sum(r['token_exact'] and not r['text_exact'] for r in selected),
                same_tokens_different_grade=sum(r['token_exact'] and r['candidate_correct']!=r['reference_correct'] for r in selected),
                correct_to_wrong=sum(r['reference_correct'] and not r['candidate_correct'] for r in selected),
                wrong_to_correct=sum(not r['reference_correct'] and r['candidate_correct'] for r in selected),
                reference_tokens=sum(r['reference_output_tokens'] for r in selected),
                candidate_tokens=sum(r['candidate_output_tokens'] for r in selected),
                reference_decode_work=sum(r['reference_decode_loop_work'] for r in selected),
                candidate_decode_work=sum(r['candidate_decode_loop_work'] for r in selected),
                scope='Common completed questions only. A matching bounded or task trace is not a numerical qualification or statistical equivalence proof.'))
    table(ROOT/'behavioral_pair_details.csv',details)
    table(ROOT/'behavioral_pair_summary.csv',summaries)
    write(ROOT/'state/behavioral_analysis.json',dict(comparisons=summaries,
        first_difference_indexing='Zero-based generated-token index. A proper-prefix difference is the first missing position.',
        gate_warning='Once outputs differ, later gate and KV differences combine backend effects with different token histories.',
        numerical_warning='Historical strict TP failure remains unchanged.'))
    print('FREE_RUNNING_PAIR_ROWS',len(details),'SUMMARIES',len(summaries),flush=True)


if __name__=='__main__':main()
