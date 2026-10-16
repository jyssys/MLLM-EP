"""Freeze teacher prefixes from prior evidence, not from this experiment's outcomes."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    questions = read(ROOT/'results/frozen_questions.json')[:8]
    old = []
    for path in sorted((PREVIOUS/'results/quality_native_v3').glob('aime24_greedy_cap8192_wave*.json')):
        old.extend(read(path))
    values = []
    for row in questions:
        match = next(r for r in old if r['problem_id']==row['id'])
        tokens = match['generated_ids']
        assert tokens[0] == 3
        removed = 2 if tokens[:2] == [3,198] else 1
        continuation = tokens[removed:]
        assert len(continuation) >= 1025
        values.append(dict(id=row['id'], prompt_ids=row['arms']['P1']['ids'],
            teacher_ids=continuation[:1025], removed_prior_tokens=tokens[:removed],
            source='Prior cap8192 greedy output, remove initial think marker and optional newline. Teacher text is diagnostic, not natural output.',
            split='calibration' if len(values)<4 else 'confirmation'))
    path = ROOT/'results/fidelity_inputs.json'
    assert not path.exists()
    write(path, values)
    print(sha(path))

if __name__=='__main__':
    main()
