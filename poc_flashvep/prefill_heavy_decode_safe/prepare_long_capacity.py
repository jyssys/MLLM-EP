"""Extend calibration only where the measured service balance justifies it."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for model,length in (('ouro',8192),('ouro',12000),('ouro',16384),('huginn',3900)):
        parent=ROOT/'configs'/f'capacity_{model}_L{8192 if model=="ouro" else 3500}_B124_start1.json'
        cfg=json.loads(parent.read_text());dataset=f'{"calibration" if length==8192 else "natural_calibration"}_{model}_L{length}.json'
        pool=json.loads((ROOT/'data'/dataset).read_text());name=f'capacity_{model}_L{length}_O16_B124_start1'
        cfg.update(name=name,length=length,cap=16,request_caps=[16]*4,state_capacity_tokens=length,chunk=length,
            token_budget=4*length,P_prefill_graph_rows=4*length,D_prefill_graph_rows=length,dataset=dataset,
            prompt_ids=[r['prompt_id'] for r in pool[:4]],jobs=[dict(batch=b,cap=16,repeats=3) for b in (1,2,4)],
            phase='LONG_CONTEXT_FINITE_CAPACITY_REFERENCE',timeout=1200,
            selection_reason='Measure whether natural long prompts make P service longer than prepared Output16 Decode. This is a hypothesis, not an idle-capacity result.')
        assert length+16<=cfg['context_limit'];save(cfg);jobs.append(name)
    write(ROOT/'state/long_capacity_plan.json',dict(jobs=jobs,
        scope='No model or attention changes. Original document prefixes. Output16. Same legal KV pools. Blocked B4 remains visible.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
