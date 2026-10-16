"""Add metadata-ordered MATH teacher cases. Keep the original eight AIME probes."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    old=[]
    for path in sorted((PREVIOUS/'results/quality_native_v3').glob('math100_greedy_cap8192_wave*.json')):
        old.extend(read(path))
    records=[]
    for row in read(ROOT/'results/fidelity_completion_inputs.json'):
        previous=next(r for r in old if r['problem_id']==row['id'])
        tokens=previous['generated_ids']
        removed=2 if tokens[:2]==[3,198] else int(tokens[0]==3)
        assert len(tokens)-removed>=1025
        records.append(row|dict(teacher_ids=tokens[removed:removed+1025],split='confirmation',
            source='Prior frozen MATH greedy output. Metadata order, not a new result-based selection.'))
    path=ROOT/'results/fidelity_additional_inputs.json'
    assert not path.exists()
    write(path,records)
    write(ROOT/'fidelity_diversity_lock.json',dict(reason='Include MATH in teacher-state checks as the specification requires.',
        inputs_sha256=sha(path),original_eight_AIME_inputs_unchanged=True,physical_HF_gpu=5,
        timing_scope='Reference state collection only. Concurrent GPU4 reference work is not a serving benchmark.'))

if __name__=='__main__':
    main()
