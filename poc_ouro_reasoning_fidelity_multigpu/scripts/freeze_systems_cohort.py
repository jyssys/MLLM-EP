"""Select real-workload inputs from metadata before new quality results."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    questions=read(ROOT/'results/frozen_questions.json')
    aime=[r for r in questions if r['dataset']=='aime24'][8:16]
    math=[r for r in questions if r['dataset']=='math100'][:30]
    requests=[]
    for row in aime+math:
        requests.append(dict(id=row['id'],dataset=row['dataset'],prompt_ids=row['arms']['P1']['ids'],
            prompt_sha256=row['arms']['P1']['text_sha256'],answer=row['answer'],seed=SEEDS[0],
            temperature=1. if row['dataset']=='aime24' else 0.,
            top_p=.7 if row['dataset']=='aime24' else 1.,top_k=-1))
    path=ROOT/'results/frozen_systems_requests.json'
    assert not path.exists()
    write(path,requests)
    concurrency_ids=[r['id'] for r in aime[:2]+math[:6]]
    write(ROOT/'systems_plan_lock.json',dict(
        conditional_gate='Credible quality and no unexplained consequential runtime bug.',
        input_sha256=sha(path),B1_counts=dict(aime24=8,math100=30),
        input_rule='AIME metadata positions 8:16, MATH metadata positions 0:30. No new quality outcome selection.',
        concurrency_ids=concurrency_ids,offered_concurrency=[1,2,4,8],
        layouts=['TP1','TP2','TP4','TP1xDP4','TP2xDP2'],
        natural_repeats=1,
        uncertainty='Problem bootstrap for independent requests. Single finite-cohort throughput is descriptive, not a repeated-cohort CI.',
        cap_rule='Use the largest complete quality cap executed for each dataset lane, without choosing by accuracy.',
        memory_policy='Record actual KV pools, occupancy, preemption, and completion. Do not infer residency from offered concurrency.',
        replay_rule='Predesignated first two AIME and first four MATH system cases. Reuse their TP4 natural output IDs, capped to first 2048 only for hardware replay.',
        historical_control='Predesignated prior coding inputs, 16K prompt and 64 forced outputs, fresh measurements only.',
        failure_rule='Keep failed/censored requests. Do not replace them with faster completed requests.',
        method_implementation=False))

if __name__=='__main__':
    main()
