"""Validate service recovery without changing any SLO threshold."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .residual_guard_study import calibrate
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .stages import state_gate
from .analyze import records,metrics
from .prediction_validation import analyze


def options(wait):
    value=method_settings(wait=wait);margin=calibrate()['total_margin_ms']
    value.update(async_decode=True,separate_nominal_objective=True,
        cost_model='coalesced_cost_model_v2.json',pipeline_objective_excludes_existing_wait=True,
        queue_benefit_external_only=True,safety_margin_ms=margin,
        safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)},
        expired_ttft_policy='QUEUE_RECOVERY',phase='EXPIRED_TTFT_RECOVERY_DISCOVERY_NOT_HELD_OUT')
    return value


def one(wait,rate,seed):
    name=f'recovery_wait{int(wait)}_DIRECT_O16_q{rate}_s{seed}_W600_v1'
    cfg=duration_config(name,'DIRECT',rate,seed=seed,warm=60.,measure=600.,follow=90.,**options(wait))
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    write(ROOT/'analysis'/(name+'_prediction.json'),analyze(name))
    print('EXPIRED_DEADLINE_RECOVERY',name,value['raw_completed_output_tokens_s'],
        {k:(v['qualified'],v['arrival_cohort_joint_attainment']) for k,v in value['SLO_regimes'].items()},flush=True)
    return value


def main():
    try:
        name='qual_expired_ttft_recovery_all_arrived_N64_O16_v1'
        cfg=base(name,'DIRECT',count=64,cap=16,rate=3.,arrival_offsets=[0.]*64,arrival_rate=None,
            verify_layer_execution=True,correctness_only=True,clean_timing=False,**options(True))
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        work=state_gate(name,64);assert work['passed']
        decisions=records(ROOT/'raw'/name/'online_decisions.jsonl');exercised=[]
        for row in decisions:
            choice=row['choice']
            if not choice['layers']:continue
            candidate=next(c for c in choice['candidates'] if c['layers']==choice['layers'])
            if candidate['expired_own_TTFT_not_protected'] or candidate['expired_pending_TTFT_not_protected']:
                exercised.append(row)
        assert exercised,'The qualification did not exercise the recovery branch'
        write(ROOT/'analysis/expired_TTFT_recovery_work_gate.json',dict(name=name,passed=True,
            work=work,exercised_recovery_decisions=exercised,evidence='CORRECTNESS_ONLY',
            scope='All64 arrivals at time0 are a branch-qualification workload, not a serving success. Past TTFT deadlines remain failed SLOs. No request is dropped, restarted or approximated. Existing ITL and future TTFT predicates stay strict.'))
        streaming=one(False,4.8,2026111458)
        whole=one(True,5.,2026111459)
        if whole['SLO_regimes']['Decode_priority']['qualified']:one(True,5.4,2026111459)
        elif streaming['SLO_regimes']['Decode_priority']['qualified']:one(False,5.2,2026111458)
        else:one(True,4.6,2026111459)
        write(ROOT/'state/expired_TTFT_recovery_discovery_complete.json',dict(time=time.time()))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
