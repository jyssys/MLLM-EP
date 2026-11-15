"""Test fixed-output bursts and real-prompt mixed-output finite cohorts."""
import random,fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .workloads import modulated_arrivals
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def one(arm,kind,layers=12,fraction=1.,rate=2.0,count=256):
    mixed=kind=='MIXED_OUTPUT';cap=200 if mixed else 16
    name=f'mixed_discovery_{kind}_{arm}_L{layers}_F{fraction}_q{rate}_N{count}_v1'
    options=method_settings()
    options.update(cost_model='coalesced_cost_model_v2.json',separate_nominal_objective=True,
        static_layers=layers,migration_fraction=fraction,phase='CONTROLLED_HETEROGENEOUS_BURST_DISCOVERY')
    if arm=='Colocated':
        # This is an objective-specific discovery control. The final baseline
        # must still be selected from the entire qualified frontier.
        options=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,
            ar_async_scheduling=True,chunk=8192,token_budget=8192,D_slots=1,
            phase='CONTROLLED_HETEROGENEOUS_BURST_DISCOVERY')
    cfg=base(name,arm,count=count,cap=cap,rate=rate,**options)
    if mixed:
        rng=random.Random(2026111461)
        cfg['request_caps']=[rng.choices([16,64,200],[.6,.3,.1])[0] for _ in range(count)]
        cfg['output_distribution']=dict(caps=[16,64,200],probabilities=[.6,.3,.1],
            scope='Controlled request caps, not Natural EOS.')
    else:
        values=modulated_arrivals(rate,10*count/rate,2026111461)
        assert len(values)>=count
        cfg['arrival_offsets']=values[:count]
        cfg['arrival_distribution']=dict(kind='PIECEWISE_POISSON',period_s=40.,
            high_s=8.,high_multiplier=3.,low_multiplier=.5,
            scope='Controlled burst trace, not a public production arrival trace.')
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
    print('MIXED_DISCOVERY',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)


def main():
    gate=ROOT/'analysis/nominal_policy_online_gate.json';limit=time.time()+10800
    while not gate.exists():
        assert time.time()<limit;time.sleep(10)
    assert json.loads(gate.read_text())['passed']
    try:
        for arm,layers,fraction in (('NATIVE',0,0.),('STATIC',12,1.),('STATIC',12,.5),
                ('STATIC',8,1.),('DIRECT',0,1.),('Colocated',0,0.)):
            one(arm,'BURSTY_O16',layers,fraction)
        for arm,layers in (('NATIVE',0),('STATIC',12),('STATIC',4),('DIRECT',0),('Colocated',0)):
            one(arm,'MIXED_OUTPUT',layers,rate=1.,count=128)
        write(ROOT/'state/mixed_discovery_complete.json',dict(time=time.time(),evidence='MEASURED',
            scope='Finite-cohort discovery only. Independent held-out and matched-budget controls remain required.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
