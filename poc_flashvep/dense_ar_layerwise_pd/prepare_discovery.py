"""Select a first workload from measured Native local capacities."""
import json,random,statistics
from .common import *
from .prepare import base,save


def main():
    initialize();assert json.loads((ROOT/'analysis/native_N32_qualification.json').read_text())['passed']
    source=ATOMIC/'data/test_qwen8_L8000.json'
    rows=json.loads(source.read_text());assert len(rows)>=128
    rows=rows[:128];assert len({r['prompt_id'] for r in rows})==128
    target=ROOT/'data/discovery_qwen8_L8000.json'
    if target.exists():assert json.loads(target.read_text())==rows
    else:write(target,rows)
    write(ROOT/'data/discovery_manifest.json',dict(source=str(source),source_SHA256=digest(source),
        target_SHA256=digest(target),distinct_prompt_ids=128,input_tokens=8000,
        scope='Preserved real controlled ShareGPT/LongBench prefixes. No repeated filler tokens.'))
    values=[json.loads(s) for s in (ROOT/'raw/capacity_qwen8_L8000_O16_p_only_B124_s0/cohorts.jsonl').read_text().splitlines()]
    capacity=max(statistics.mean(r['batch']/r['P_plus_first_sample_s'] for r in values if r['batch']==b) for b in (1,2,4))
    jobs=[]
    for factor in (1.1,1.3):
        rate=capacity*factor;rng=random.Random(2026111203);offsets=[0.]
        for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
        name=f'native_qwen8_L8000_O16_N64_load{int(factor*100)}_s3'
        c=base(name,count=64,cap=16);c.update(dataset=target.name,arrival_rate=rate,arrival_offsets=offsets,
            arrival_seed=2026111203,phase='DISCOVERY_PILOT',clean_timing=True,timeout=180)
        jobs.append(save(c))
    write(ROOT/'state/discovery_plan.json',dict(jobs=jobs,P_capacity_requests_s=capacity,
        scope='Independent finite P service rate selects pilot QPS. It is not online capacity or same-window headroom.'))
    print(json.dumps(jobs))


if __name__=='__main__':main()
