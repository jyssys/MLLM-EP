"""Fix the mixture arrival rate only after its finite calibration completes."""
import json,random,time,fcntl
from .common import *
from .capacity_analysis import analyze
from .run import execute


def main():
    initialize();plan=json.loads((ROOT/'state/mixed_native_discovery_plan.json').read_text())
    while True:
        ready=True
        for name in (plan['capacity'],plan['state8']):
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists():ready=False;continue
            value=json.loads(receipt.read_text())
            assert value.get('complete'),('Mixed predecessor failed',name,value)
        qualification=ROOT/'raw'/plan['state8']/'qualification_with_frozen_reference.json'
        if ready and qualification.exists():
            assert json.loads(qualification.read_text())['campaign_gate_passed'];break
        assert time.time()<LONG_START_CUTOFF
        time.sleep(10)
    best=analyze(plan['capacity'])['strongest_finite_P_reference'];assert best
    rate=1.15*best['P_requests_per_s']['p50'];seed=2026111250;rng=random.Random(seed);offsets=[0.]
    for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
    cfg=json.loads((ROOT/'configs'/'reserve_huginn_L3900_O16_P2_mu115_N64_a2026111230.json').read_text())
    data=json.loads((ROOT/'data'/'mixed_discovery_huginn_L3900_L1024_p50.json').read_text())
    name='mixed_huginn_L3900_L1024_p50_P2_mu115_N64_a2026111250'
    cfg.update(name=name,count=64,dataset='mixed_discovery_huginn_L3900_L1024_p50.json',
        prompt_ids=[r['prompt_id'] for r in data[:64]],
        prompt_token_lengths=[len(r['input_ids']['huginn']) for r in data[:64]],
        arrival_seed=seed,arrival_rate=rate,arrival_offsets=offsets,
        capacity_reference=plan['capacity'],required_qualification=plan['state8'],comparison_reference=None,
        phase='MIXED_REAL_PREFIX_NATIVE_DISCOVERY',
        selection_reason='Measure whether the long/short reservation mixture permits a growing compute-ready queue and genuine D opportunities. This is not the frozen homogeneous workload.')
    save(cfg);write(ROOT/'state/mixed_native_pilot_plan.json',dict(jobs=[name],capacity=best,
        offered_QPS=rate,scope='New Poisson discovery trace. Not a held-out confirmation or Phase 2 migration.'))
    ok=execute(name);write(ROOT/'state/mixed_native_pilot_receipt.json',dict(name=name,passed=ok))
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources()
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not owner.processes():owner.restore(list(range(8)))


if __name__=='__main__':main()
