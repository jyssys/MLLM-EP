"""Prepare qualified engine controls and the native dense-AR diagnostic."""
import argparse
import random
import statistics
from .common import *
from .prepare import base,GRID
from .phase_b_configs import pd_config
from .summary_tables import records,percentile


def save(cfg):
    path=ROOT/'configs'/(cfg['name']+'.json')
    assert not path.exists(),'Preserve the existing configuration'
    write(path,cfg)
    return path.name


def enqueue(names):
    path=ROOT/'configs/phase_b_screen_plan.json'
    old=json.loads(path.read_text())
    archive=ROOT/'configs/phase_b_screen_initial_plan.json'
    if not archive.exists():write(archive,old)
    write(path,dict(configs=list(dict.fromkeys(names+old['configs']))))
    append(ROOT/'state/plan_extensions.jsonl',dict(time=time.time(),names=names,
        reason='Qualify generic engine controls before testing a loop-specific claim. Preserve the original grid.'))


def prepare():
    names=[]
    c=dict(name='calibration_qwen_g2_start0',model='qwen',worker='ar_pd',gpus=[2,3],count=8,
        dataset='calibration_L3072.json',seed=2026103107,arrival_offsets=[0.]*8,request_caps=[128]*8,
        estimated_seconds=180,timeout=600,phase='calibration',workload='G2',calibration_serial=True,
        supported_chunked_prefill=True)
    names.append(save(c))
    for model,count,cell,pool,slots in (('ouro',24,'G3',22528,32),('huginn',4,'G2',26624,8)):
        length,cap=GRID[cell]
        native_name=f'qualification_{model}_expanded_native_v1'
        c=base(model,native_name)
        c.update(native_reference_only=True,state_probe=True,slots=slots,capacity=slots,
            chunk=2048,kv_blocks=pool,phase='engine_correctness',estimated_seconds=300,
            groups=[dict(chunk=2048,kv_blocks=pool,jobs=[dict(cell=cell,dataset=f'test_L{length}.json',batch=count,cap=cap)])])
        names.append(save(c))
        c=pd_config(model,f'qualification_{model}_expanded_pd_v1',cell,count,100.)
        c.update(capacity=slots,slots=slots,chunk=2048,token_budget=2048,kv_blocks=pool,
            arrival_offsets=[0.]*count,verify_transfer=True,state_probe=True,
            phase='engine_correctness',estimated_seconds=300,qualification_native=native_name)
        names.append(save(c))
    enqueue(names)


def freeze_qwen():
    folder=ROOT/'raw/calibration_qwen_g2_start0'
    assert json.loads((folder/'status.json').read_text())['status']=='COMPLETE'
    rows=records(folder/'requests.jsonl');assert len(rows)==8
    values=[1000*(b['time']-a['time']) for r in rows for a,b in zip(r['token_events'],r['token_events'][1:]) if b['count']-a['count']==1]
    path=ROOT/'configs/frozen_SLO_qwen.json';assert not path.exists()
    write(path,dict(qwen=dict(TTFT_ms=2*percentile([r['ttft_ms'] for r in rows],.95),
        TPOT_ms=2*percentile([r['tpot_ms'] for r in rows],.95),ITL_ms=2*percentile(values,.99),
        calibration_requests=8,calibration_run=folder.name,time=time.time(),
        rule='2x serial calibration p95 TTFT / request TPOT, 2x population p99 ITL',
        evidence='MEASURED_CALIBRATION_FROZEN_BEFORE_HELD_OUT')))


def add_profiles():
    names=[]
    c=base('huginn','phase_a_huginn_expanded_tuned_start0')
    c.update(chunk=2048,kv_blocks=26624,phase='capacity',estimated_seconds=300,
        canonical_reference='phase_a_huginn_expanded_pool_start0',
        groups=[dict(chunk=2048,kv_blocks=26624,jobs=[dict(cell='G2',dataset='test_L3072.json',batch=4,cap=128,repeats=3)])])
    names.append(save(c))
    for model,cell in (('ouro','G1'),('huginn','G2')):
        old=json.loads((ROOT/'configs'/f'phase_b_{model}_{cell.lower()}_screen_start0.json').read_text())
        c=dict(old,name=f'profile_{model}_native_pd_{cell.lower()}_v1',count=8,
            request_caps=old['request_caps'][:8],arrival_offsets=old['arrival_offsets'][:8],
            prompt_ids=old['prompt_ids'][:8],phase='profile',profile=True,
            profile_schedule=dict(wait=200,active=200),estimated_seconds=300)
        names.append(save(c))
    enqueue(names)


def qualify():
    from .gates import qualification
    for model,count in (('ouro',24),('huginn',4)):
        qualification(model,count,f'qualification_{model}_expanded_pd_v1',f'qualification_{model}_expanded_native_v1')


def controls():
    names=[]
    for model,count in (('ouro',24),('huginn',4)):
        gate=json.loads((ROOT/'state'/f'{model}_qualification{count}.json').read_text())
        assert gate['status']==f'PASS_{count}_NATIVE_PD_QUALIFICATION'
    for model in ('ouro','huginn'):
        for cell in ('G2','G3'):
            old=json.loads((ROOT/'configs'/f'phase_b_{model}_{cell.lower()}_screen_start0.json').read_text())
            c=dict(old,name=f'phase_b_{model}_{cell.lower()}_tuned_chunk_start0',chunk=2048,token_budget=2048,
                phase='focused_crossover',protocol_control='Same native NIXL and arrival trace. Only the generic chunk budget changes.')
            names.append(save(c))
    # These are finite service measurements, not a prediction of stable serving capacity.
    for model,cell,pool,slots,batch in (('ouro','G3',22528,32,24),('huginn','G2',26624,8,4)):
        from .b_capacity_plan import capacity
        measured=capacity(model,cell,chunk=2048,pool=pool)
        if model=='ouro':
            prepared=json.loads((ROOT/'raw/phase_a_ouro_native_pd_donly_b24_start0/D_only_window.json').read_text())
            measured['mu_D']=min(measured['mu_D'],prepared['mu_D_finite'])
            measured['prepared_native_PD_D_evidence']=prepared
        for start in range(3):
            count=64;rate=.8*min(measured['mu_P'],measured['mu_D'])
            c=pd_config(model,f'phase_b_{model}_{cell.lower()}_expanded_start{start}',cell,count,rate)
            seed=2026103111+start; rng=random.Random(seed);offsets=[0.]
            for _ in range(count-1):offsets.append(offsets[-1]+rng.expovariate(rate))
            c.update(chunk=2048,token_budget=2048,kv_blocks=pool,slots=slots,capacity=slots,
                dataset=f'test64_L{GRID[cell][0]}.json',arrival_seed=seed,arrival_offsets=offsets,
                gpus=([0,1] if start%2==0 else [2,3]) if model=='ouro' else ([4,5] if start%2==0 else [6,7]),
                capacity_planning=measured,start=start,rate_factor=.8,phase='focused_crossover',
                estimated_seconds=600,engine_control='Expanded KV pool and compute slots are generic engine tuning. Not a loop algorithm.')
            names.append(save(c))
    assert (ROOT/'configs/frozen_SLO_qwen.json').exists()
    all_rows=[]
    for run in ('phase_a_qwen_grid_start0','phase_a_qwen_large_batch_start0'):
        all_rows+=records(ROOT/'raw'/run/'cohorts.jsonl')
    for cell in ('G1','G2','G3','G4','G5','G6','C1','C2'):
        rows=[r for r in all_rows if r['job']['cell']==cell]
        batches=sorted({r['batch'] for r in rows})
        muP=max(b/statistics.median(r['prefill_plus_first_token_service_s'] for r in rows if r['batch']==b) for b in batches)
        muD=max(b/statistics.median(r['decode_rest_service_s'] for r in rows if r['batch']==b) for b in batches)
        rate=.8*min(muP,muD);count=64;seed=2026103111;rng=random.Random(seed);offsets=[0.]
        for _ in range(count-1):offsets.append(offsets[-1]+rng.expovariate(rate))
        c=dict(name=f'phase_b_qwen_{cell.lower()}_screen_start0',model='qwen',worker='ar_pd',gpus=[2,3],
            count=count,dataset=f'test64_L{GRID[cell][0]}.json',seed=2026103107,arrival_seed=seed,
            arrival_offsets=offsets,request_caps=[GRID[cell][1]]*count,arrival_rate=rate,
            estimated_seconds=180,timeout=600,phase='crossover',workload=cell,start=0,
            capacity_planning=dict(mu_P=muP,mu_D=muD,scope='Native vLLM finite independent service rates. Engine-confounded diagnostic.'),
            supported_chunked_prefill=True)
        names.append(save(c))
    enqueue(names)


def focused():
    from .b_capacity_plan import capacity
    names=[]
    for model,pool in (('ouro',8192),('huginn',18432)):
        measured=capacity(model,'G2',pool=pool)
        prepared=[]
        for path in (ROOT/'raw').glob(f'phase_a_{model}_native_pd_donly*/D_only_window.json'):
            cfg=json.loads((ROOT/'configs'/(path.parent.name+'.json')).read_text())
            if cfg['dataset']=='test_L3072.json' and cfg['kv_blocks']==pool:
                prepared.append(json.loads(path.read_text())['mu_D_finite'])
        if prepared:
            measured['mu_D']=min(measured['mu_D'],max(prepared))
            measured['actual_prepared_native_PD_D_rates']=prepared
        for factor,start in [(v,0) for v in (.4,.6,.8,1.,1.2)]+[(.8,1),(.8,2)]:
            seed=2026103111+start;rate=factor*min(measured['mu_P'],measured['mu_D']);count=64
            tag=str(factor).replace('.','p');c=pd_config(model,f'phase_b_{model}_g2_rate{tag}_start{start}','G2',count,rate)
            rng=random.Random(seed);offsets=[0.]
            for _ in range(count-1):offsets.append(offsets[-1]+rng.expovariate(rate))
            c.update(dataset='test64_L3072.json',arrival_seed=seed,arrival_offsets=offsets,start=start,
                gpus=([0,1] if start%2==0 else [2,3]) if model=='ouro' else ([4,5] if start%2==0 else [6,7]),
                capacity_planning=measured,rate_factor=factor,phase='focused_crossover',estimated_seconds=1200)
            names.append(save(c))
        old=json.loads((ROOT/'configs'/f'phase_b_{model}_g2_screen_start0.json').read_text())
        for arm,chunk,wavefront in (('DP',4096,False),('DP_Chunked',2048,True)):
            c=dict(old,name=f'phase_b_{model}_g2_{arm.lower()}_start0',arm=arm,chunk=chunk,token_budget=chunk,
                wavefront=wavefront,phase='repeat',reference_run=old['name'],
                baseline_scope='Two independent colocated replicas. Shared-core Mixed is disabled. Not a production Sarathi implementation.')
            names.append(save(c))
    path=ROOT/'configs/phase_b_screen_plan.json';plan=json.loads(path.read_text())
    write(path,dict(configs=list(dict.fromkeys(plan['configs']+names))))
    append(ROOT/'state/plan_extensions.jsonl',dict(time=time.time(),names=names,
        reason='Complete the predeclared five-rate G2 sweep and three independent starts. Keep both models symmetric.'))


def main():
    actions=dict(prepare=prepare,freeze_qwen=freeze_qwen,qualify=qualify,controls=controls,add_profiles=add_profiles,focused=focused)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=tuple(actions))
    actions[parser.parse_args().action]()


if __name__=='__main__':main()
