"""Test a residual guard calibrated only from completed discovery runs."""
import fcntl,math
from .common import *
from .prediction_validation import analyze
from .method_discovery_v2 import method_settings
from .configuration import base
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def calibrate():
    path=ROOT/'configs/discovery_residual_guard_v1.json'
    if path.exists():return json.loads(path.read_text())
    names=['async_D_DIRECT_nominal_N128_O16_q4.8_v1',
        'rate_method3_DIRECT_L12_O16_q4.0_s2026111453_W600.0_generic_async_discovery_v1_ASYNC_D']
    values=[analyze(name) for name in names]
    maximum=max(v['errors']['group_first_output_upper']['maximum_underprediction_ms'] for v in values)
    increment=10.*math.ceil((maximum+10.)/10.)
    result=dict(time=time.time(),calibration_names=names,base_margin_ms=110.,
        maximum_observed_upper_residual_ms=maximum,additional_margin_ms=increment,
        total_margin_ms=110.+increment,evidence='TRACE_ANALYSIS',
        scope='Use the largest positive group residual plus10ms, rounded upward to10ms. This is a discovery envelope, not an independent prediction guarantee or a WCET. Groups within a start are dependent. Runtime retirement guards remain unchanged.')
    write(path,result);return result


def main():
    guard=calibrate();margin=guard['total_margin_ms'];options=method_settings()
    options.update(async_decode=True,separate_nominal_objective=True,
        cost_model='coalesced_cost_model_v2.json',safety_margin_ms=margin,
        safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)},
        phase='RESIDUAL_GUARD_CALIBRATION_VALIDATION_NOT_HELD_OUT')
    try:
        name=f'residual_guard_{int(margin)}_DIRECT_N128_O16_q4.8_v1'
        cfg=base(name,'DIRECT',count=128,cap=16,rate=4.8,**options)
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
        write(ROOT/'analysis'/(name+'_prediction.json'),analyze(name))
        print('RESIDUAL_GUARD',margin,value['TTFT_p95_ms'],value['ITL_p99_ms'],flush=True)
        name=f'residual_guard_{int(margin)}_DIRECT_O16_q4.0_s2026111457_W600_v1'
        cfg=duration_config(name,'DIRECT',4.,seed=2026111457,warm=60.,measure=600.,follow=90.,**options)
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
        write(ROOT/'analysis'/(name+'_prediction.json'),analyze(name))
        write(ROOT/'analysis/residual_guard_study_complete.json',dict(name=name,margin_ms=margin,
            qualified={k:v['qualified'] for k,v in value['SLO_regimes'].items()},time=time.time()))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
