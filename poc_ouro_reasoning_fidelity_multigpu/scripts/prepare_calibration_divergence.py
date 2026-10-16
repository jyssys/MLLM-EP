"""Minimize natural-output differences using calibration questions only."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    records=[]
    for row in read(ROOT/'results/fidelity_inputs.json')[:4]:
        assert row['split']=='calibration'
        native=read(ROOT/f'results/native_fidelity_TP1_v1/natural_{row["id"]}_cap256_repeat0.json')
        reference=read(ROOT/f'results/hf457_fidelity_v1/natural_{row["id"]}_cap256.json')
        divergence=next((i for i,(a,b) in enumerate(zip(native['ids'],reference['ids'])) if a!=b),None)
        if divergence is None:continue
        assert native['ids'][:divergence]==reference['ids'][:divergence]
        records.append(dict(id=row['id'],split='calibration',prompt_ids=row['prompt_ids'],step=divergence,
            teacher_ids=reference['ids'][:divergence+1],reference_next=reference['ids'][divergence],
            prior_native_next=native['ids'][divergence],
            source='First natural greedy divergence on the four predefined calibration questions. No confirmation outcome used.'))
    path=ROOT/'results/calibration_divergent_prefixes.json'
    assert not path.exists()
    write(path,records)
    print([(r['id'],r['step']) for r in records])

if __name__=='__main__':
    main()
