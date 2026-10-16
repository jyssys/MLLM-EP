"""Compare the predeclared first batch across two identical engine configurations."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    pilot=ROOT/'results/prompt_diagnostic_v1'
    full=ROOT/'results/quality_v1'
    a=read(pilot/'diagnostic_P1_cap8192_wave00.json')
    b=read(full/'aime24_sampling_cap8192_wave00.json')
    assert read(pilot/'engine_contract.json')==read(full/'engine_contract.json')
    rows=[]
    for x,y in zip(a,b):
        assert (x['problem_id'],x['seed'],x['cap'],x['prompt_sha256'])==(y['problem_id'],y['seed'],y['cap'],y['prompt_sha256'])
        rows.append(dict(id=x['problem_id'],seed=x['seed'],cap=x['cap'],
            exact_equal=x['generated_ids']==y['generated_ids'],
            first_divergence=next((i for i,(u,v) in enumerate(zip(x['generated_ids'],y['generated_ids'])) if u!=v),None),
            pilot_tokens=x['generated_tokens'],full_tokens=y['generated_tokens']))
    target=ROOT/'results/quality_engine_restart_reproducibility.json'
    assert not target.exists()
    write(target,dict(engine_kwargs_identical=True,comparisons=rows,
        all_equal=all(r['exact_equal'] for r in rows),
        scope='Four predefined first-batch cases across fresh engine sessions. Not a universal determinism or accuracy claim.'))
    print(rows)

if __name__=='__main__':
    main()
