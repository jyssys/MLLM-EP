"""Screen CUDA-event enclosure interference against two marker-free starts."""
import fcntl
import json
import time
from .common import ROOT, QUIET_LOCK, resources, write
from .discovery import job
from .analysis import metrics


def main():
    results=[]
    try:
        for method in ('Static','Token'):
            control=f'matched_rows_{method}_Last12_P4_D4_N144_q6.4_v1'
            path=ROOT/'raw'/control/'dispatcher_result.json'
            while not path.exists() or not json.loads(path.read_text()).get('complete'):
                assert time.time()<json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-7200
                time.sleep(15)
            source=json.loads((ROOT/'configs'/(control+'.json')).read_text())
            values=[metrics(control)]
            for label,enabled in (('events',True),('control2',False)):
                cfg=dict(source,name=f'enclosure_{method}_{label}_N144_v1',
                    execution_events=enabled,boundary_events=enabled,clean_timing=not enabled,
                    phase='CUDA_EVENT_INTERFERENCE_SCREEN',correctness_only=False)
                value=job(cfg)
                assert value
                values.append(value)
            keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')
            controls={key:(values[0][key]+values[2][key])/2 for key in keys}
            ratios={key:values[1][key]/controls[key] for key in keys}
            passed=all(abs(ratio-1)<=.05 for ratio in ratios.values())
            result=dict(method=method,names=[v['name'] for v in values],threshold_fraction=.05,
                ratios_to_mean_marker_free=ratios,passed=passed,
                evidence='MEASURED_INTERFERENCE_SCREEN',
                scope='One event-capture start versus two marker-free starts at the same trace and actual geometry. A point-estimate screen is not a statistical zero-interference guarantee. CUDA events enclose stream work and gaps. They are not exclusive kernel execution or physical kernel busy/idle. Failed screens remain diagnostic.')
            results.append(result)
            write(ROOT/'analysis/CUDA_event_interference_screen.json',results)
            print('EVENT_SCREEN',method,passed,ratios,flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
