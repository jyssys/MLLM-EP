"""Grade bounded reference probes separately from the full benchmark."""
import sys
sys.path.insert(0,'/home/esjung/external/looped-exact-depth-chunk-assets/cpu_deps')
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer,aime_integer

def main():
    from math_verify import parse,verify,LatexExtractionConfig
    questions={r['id']:r for r in read(ROOT/'results/frozen_questions.json')}
    records=[]
    for folder,label in [('hf457_fidelity_v1','HF457')]+[(f'native_fidelity_TP{tp}_v1',f'TP{tp}') for tp in [1,2,4]]:
        for path in (ROOT/'results'/folder).rglob('*cap*.json'):
            row=read(path)
            if not isinstance(row,dict) or not {'id','cap','ids','text','finish_reason'}<=set(row):continue
            if row.get('repeat',0)!=0:continue
            gold=questions[row['id']]
            answer,rule=final_answer(row['text'])
            correct=False
            if gold['dataset']=='aime24':
                normalized=aime_integer(answer)
                parsed=normalized is not None
                correct=parsed and normalized==aime_integer(gold['answer'])
            else:
                parsed=False
                if answer is not None:
                    a=parse('$'+gold['answer']+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                    b=parse('$'+answer+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
                    parsed=bool(b)
                    correct=bool(verify(a,b,strict=True,timeout_seconds=2))
            records.append(dict(engine=label,id=row['id'],dataset=gold['dataset'],cap=row['cap'],
                tokens=len(row['ids']),finish_reason=row['finish_reason'],parsed=parsed,
                answer=answer,reference=gold['answer'],answer_matches=bool(correct),
                completed_correct=bool(correct and row['finish_reason']=='stop'),parse_rule=rule,
                scope='Small fidelity probe, not full AIME or MATH benchmark accuracy.'))
    write(ROOT/'results/fidelity_natural_grading.json',records)
    print([(r['engine'],r['id'],r['cap'],r['completed_correct']) for r in records if r['dataset']=='math100' and r['cap']==8192])

if __name__=='__main__':
    main()
