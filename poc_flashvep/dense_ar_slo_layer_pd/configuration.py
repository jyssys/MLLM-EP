"""Build immutable discovery jobs from the qualified exact runtime."""
import random,fcntl
from .common import *


def offsets(count,rate,seed):
    rng=random.Random(seed);values=[0.]
    for _ in range(count-1):values.append(values[-1]+rng.expovariate(rate))
    return values


def base(name,arm='Native',count=64,cap=16,rate=3.,seed=2026111421,dataset='discovery.json',**changes):
    template_arm='Revised' if arm not in ('Native','Colocated','STATIC') else 'Static' if arm=='STATIC' else arm
    template=f'heldout_{template_arm}_O16_N128_q4.8_s2026111381_v1'
    c=json.loads((PREVIOUS/'configs'/(template+'.json')).read_text())
    c.pop('selection_SHA256',None)
    c.update(name=name,count=count,cap=cap,dataset=dataset,request_caps=[cap]*count,
        arrival_offsets=offsets(count,rate,seed),arrival_rate=rate,arrival_seed=seed,
        phase='SLO_SPECIFIC_FRONTIER_DISCOVERY',evaluation_arm=arm,
        physical_activity=False,execution_events=False,boundary_events=False,
        full_prompt_state_witness=False,hidden_verify_hash=False,verify_layer_execution=False,
        clean_timing=True,timeout=1200,wall_timeout=1600,
        cost_model='revised_cost_model_v3.json',defer_completion_log=True,phase_stable_norm=True)
    if arm=='Colocated':c.update(arm='DP2',module='native',policy=None)
    elif arm=='Native':c.update(arm='PD_G1',module='native',policy=None)
    else:c.update(arm='PD_G1',module='dynamic_native',policy='REVISED' if arm=='Revised' else arm)
    c.update(changes)
    source_path=ROOT/'data'/c['dataset']
    if source_path.exists():
        pool=json.loads(source_path.read_text())
        if count>len(pool):
            assert pool,'An empty prompt pool cannot supply a request'
            dataset_name=source_path.stem+'_declared_cycle_N'+str(count)+'.json'
            expanded=[pool[i%len(pool)] for i in range(count)]
            destination=ROOT/'data'/dataset_name
            with QUIET_LOCK.open('a+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                if destination.exists():assert json.loads(destination.read_text())==expanded
                else:write(destination,expanded)
            c.update(dataset=dataset_name,prompt_reuse_scope=
                'Cycle the declared real-text pool. Disable prefix caching. Repeated prompts are not independent inference samples.')
    if c.get('verified_ByteLevel_priming'):
        c['fast_priming_contract_SHA256']=digest(ROOT/'configs/verified_ByteLevel_text_priming.json')
    return c
