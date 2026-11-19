"""Gate every serving stage on the same frozen Natural EOS output contract."""
import collections
import json
import sys
import time
import numpy as np
from .common import ROOT,write,digest,save,initialize
from .qualification import configuration,records
from .run import execute
from .metrics import summarize


GOLD='standard_HND_BI1_LongBench200_natural_C1_cap2048_v1'


def arrivals(count,rate,seed):
    values=np.cumsum(np.random.default_rng(seed).exponential(1/rate,count))
    return (values-values[0]).tolist()


def output_gate(name):
    teacher={r['prompt_id']:r for r in json.loads((ROOT/'raw'/GOLD/'natural_reference.json').read_text())['rows']}
    rows=records(ROOT/'raw'/name/'requests.jsonl');cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
    checks=[dict(prompt_id=row['prompt_id'],matched=row['output_ids']==teacher[row['prompt_id']]['output_ids'],
        actual_tokens=len(row['output_ids']),reference_tokens=len(teacher[row['prompt_id']]['output_ids'])) for row in rows]
    lifetime={p.name:json.loads(p.read_text())['passed'] for p in (ROOT/'raw'/name).glob('KV_lifetime_*.json')}
    passed=len(checks)==cfg['count'] and all(r['matched'] for r in checks) and len(lifetime)==2 and all(lifetime.values())
    write(ROOT/'analysis'/(name+'_output_gate.json'),dict(passed=passed,checks=checks,lifetime=lifetime,
        evidence='CORRECTNESS_ONLY',reference=GOLD,dataset_SHA256=cfg['dataset_SHA256']))
    assert passed,('OUTPUT_OR_LIFETIME_GATE_FAILED',name)


def serving(name,method,dataset,count,rate,seed,**extra):
    cfg=configuration(name,method,dataset,count=count,serial=False,state_capture=False,correctness_only=False,
        batch_invariant=True,arrival_offsets=arrivals(count,rate,seed),arrival_seed=seed,offered_QPS=rate,
        wall_timeout=3600,serving_timeout=2700,**extra)
    assert execute(save(cfg)),name
    output_gate(name)
    return summarize(name)


def main():
    initialize();gate=ROOT/'analysis/official_invariant_N8_qualification_v1.json'
    while not gate.exists():time.sleep(2)
    assert json.loads(gate.read_text())['passed'],'Do not start serving after an invariant N8 failure'
    dataset='longbench_v1_mixed200.json'
    cfg=dict(name=GOLD,method='Native',entry_module='poc_flashvep.dense_ar_native_reproduction.natural_reference',
        gpus=[1],batch_invariant=True,kv_layout='HND',token_budget=8192,max_num_seqs=1024,context_limit=40960,
        output_cap=2048,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),
        execution_family='STOCK_WITH_OFFICIAL_BATCH_INVARIANCE',wall_timeout=10800,seed=2026111901)
    assert execute(save(cfg)),GOLD
    rows=json.loads((ROOT/'data'/dataset).read_text())
    subset=ROOT/'data/qualification32.json'
    if subset.exists():assert json.loads(subset.read_text())==rows[:32]
    else:write(subset,rows[:32])
    for method in ('Native','Agg'):
        serving(f'qual_{method}_LongBench_N32_B8192_S1024_BI1_q2_v1',method,subset.name,32,2.,2026112011,
            phase='CONTINUOUS_N32_CORRECTNESS_ONLY')
    pilot=[]
    subset=ROOT/'data/capacity_pilot40.json'
    selected=[]
    per_task=collections.Counter()
    for row in rows:
        if per_task[row['task']]<8:selected.append(row);per_task[row['task']]+=1
    assert len(selected)==40 and set(per_task.values())=={8}
    if subset.exists():assert json.loads(subset.read_text())==selected
    else:write(subset,selected)
    for rate in (.5,2.,4.):
        for method in ('Native','Agg'):
            name=f'pilot_{method}_LongBench_N40_q{rate:g}_BI1_v1'
            pilot.append(serving(name,method,subset.name,40,rate,2026112012,phase='CAPACITY_PILOT_NOT_MAIN'))
    # A finite-cohort pilot estimates the knee. It does not prove sustainable capacity.
    capacity={method:max(r['completed_requests_per_second'] for r in pilot if r['method']==method) for method in ('Native','Agg')}
    low=min(capacity.values());high=max(capacity.values())
    grid=sorted(set(round(v*20)/20 for v in (.4*low,.7*low,low,1.2*low,high,1.4*high,2*high)))
    grid=[max(.1,x) for x in grid]
    protocol=ROOT/'configs/frozen_LongBench200_Main_grid_v1.json'
    design=dict(time=time.time(),grid=grid,important_QPS=[grid[2],grid[-2]],
        capacity_pilot=capacity,knee_estimate_only=True,selection_rule='Round0.05: 0.4/0.7/1.0/1.2 times lower pilot completed rate plus1.0/1.4/2.0 times higher rate.',
        dataset_SHA256=digest(ROOT/'data'/dataset),numerical_family='OFFICIAL_BATCH_INVARIANT',
        Native_protocol='OFFICIAL_PROXY',primary_B_S=[8192,1024],SLO=dict(TTFT_ms=5000,every_ITL_ms=34,TPOT_ms=33),
        seeds=[2026112021,2026112022,2026112023],scope='Common QPS selection from pilot. Frozen before Main. No Layer/Token. Finite-cohort throughput is not sustainable capacity.')
    if protocol.exists():
        frozen=json.loads(protocol.read_text());grid=frozen['grid'];design=frozen
    else:write(protocol,design)
    write(ROOT/'state/capacity_pilot_complete.json',dict(time=time.time(),complete=True,
        Main_status='NOT_RUN_PENDING_NATIVE_AND_BS_QUALIFICATION',capacity_estimates=capacity))
    if '--main' not in sys.argv:return
    for file in ('Native_reproduction_pre_main_gate.json','B_S_pre_main_gate.json'):
        assert json.loads((ROOT/'analysis'/file).read_text())['passed'],file
    for start,seed in enumerate(design['seeds']):
        for rate in grid if start==0 else design['important_QPS']:
            order=('Native','Agg') if start%2==0 else ('Agg','Native')
            for method in order:
                serving(f'Main_{method}_LongBench_N200_q{rate:g}_s{seed}_BI1_v1',method,dataset,200,rate,seed,phase='MAIN_COMMON_B_S')
        from .figures import main as figures
        figures()
    write(ROOT/'state/main_common_sweep_complete.json',dict(time=time.time(),complete=True))
    figures()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/campaign_hold.json',dict(time=time.time(),error=repr(error),
            reason='Stop downstream dispatch after any correctness, resource or execution gate failure. Preserve all evidence.'))
        raise
