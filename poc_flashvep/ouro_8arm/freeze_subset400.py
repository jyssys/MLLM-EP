"""User-approved 400-question target. Reuse the pilot and sample only metadata."""
import random
from .contract import *


def main():
    target=ROOT/'evaluation_subset400.json';assert not target.exists()
    rows=dataset(False);pilot=dataset(True);rng=random.Random(20261021)
    pilot_ids={r['id'] for r in pilot};extension=[]
    for domain,n in [('gsm8k',236),('math500',50)]:
        population=sorted([r['id'] for r in rows if r['dataset']==domain and r['id'] not in pilot_ids],key=lambda s:int(s.rsplit('/',1)[1]))
        selected=rng.sample(population,n)
        extension.extend(sorted(selected,key=lambda s:int(s.rsplit('/',1)[1])))
    cohorts=[dict(index=i//4,ids=[r['id'] for r in pilot[i:i+4]],source='pilot') for i in range(0,len(pilot),4)]
    offset=len(cohorts)
    cohorts.extend(dict(index=offset+i//4,ids=extension[i:i+4],source='extension') for i in range(0,len(extension),4))
    write(target,dict(target_per_arm=400,domain_counts=dict(gsm8k=300,math500=100),seed=20261021,
        pilot_ids=[r['id'] for r in pilot],extension_ids=extension,cohorts=cohorts,
        rule='Keep every pilot request and its original cohort. Sample the other IDs uniformly without answers, depth, output length, or timing.',
        limitation='Continuity pilot plus randomized extension, not a uniform random sample of the complete test suites.',
        source_question_manifest_sha256=sha(ROOT/'dataset_manifest.json'),frozen_token_IDs_sha256=sha(ROOT/'frozen_prompt_ids.json'),
        authority='Latest user request reduces each setting from 1819 questions to approximately 400-500 and prefers faster completion.',
        unchanged='Checkpoint, BF16, prompts, greedy decoding, 4096-token cap, gate, threshold, KV policy, and numerical failure records.'))
    print('FROZEN_SUBSET',400,'=',300,'GSM8K +',100,'MATH; new',len(extension),'per arm',flush=True)


if __name__=='__main__':main()
