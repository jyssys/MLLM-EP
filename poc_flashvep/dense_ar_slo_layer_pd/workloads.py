"""Create declared mixed and bursty open-loop traces without future-aware policies."""
import random,fcntl
from .common import ROOT,write,json,digest,QUIET_LOCK
from .sustainable import duration_config


def modulated_arrivals(rate,horizon,seed):
    rng=random.Random(seed);values=[];at=0.
    # Eight seconds at3x and32 seconds at0.5x preserve the expected mean.
    while at<horizon:
        phase=at%40.;boundary=at+(8.-phase if phase<8. else 40.-phase)
        multiplier=3. if phase<8. else .5
        candidate=at+rng.expovariate(rate*multiplier)
        if candidate<min(boundary,horizon):values.append(candidate);at=candidate
        else:at=boundary
    return values


def mixed_duration(name,arm,rate,seed,bursty=False,mixed_output=True,measure=600.,**changes):
    cfg=duration_config(name,arm,rate,cap=200 if mixed_output else 16,seed=seed,
        warm=60.,measure=measure,follow=90.,**changes)
    if bursty:
        cfg['arrival_offsets']=modulated_arrivals(rate,cfg['fixed_measurement']['arrival_horizon_seconds'],seed)
    count=len(cfg['arrival_offsets']);rng=random.Random(seed+100000)
    caps=[rng.choices([16,64,200],[.6,.3,.1])[0] if mixed_output else 16 for _ in range(count)]
    cfg.update(count=count,request_caps=caps,
        workload_class='BURSTY_MIXED_OUTPUT' if bursty and mixed_output else 'BURSTY_O16' if bursty else 'MIXED_OUTPUT',
        output_distribution=dict(kind='CONTROLLED_KNOWN_REQUEST_CAPS_NOT_NATURAL_EOS',caps=[16,64,200],probabilities=[.6,.3,.1]) if mixed_output else dict(cap=16),
        arrival_distribution=dict(kind='PIECEWISE_POISSON_NOT_PUBLIC_TRACE',period_s=40.,high_s=8.,high_multiplier=3.,low_multiplier=.5) if bursty else dict(kind='POISSON'))
    pool=json.loads((ROOT/'data/discovery.json').read_text())
    dataset=f'{name}_real_prompt_cycle.json';target=ROOT/'data'/dataset
    rows=[pool[i%len(pool)] for i in range(count)]
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if target.exists():assert json.loads(target.read_text())==rows
        else:write(target,rows)
    cfg['dataset']=dataset;cfg['request_trace_SHA256']=digest(target)
    return cfg
