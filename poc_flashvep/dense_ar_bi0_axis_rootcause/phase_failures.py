"""Close an optional failed profiler phase. Do not qualify its measurements."""
import json
import time
from .common import ROOT,write,digest


def main():
    name='DKernelOnly_Token4_profile_q2.35_s2026112061_N200_v1';folder=ROOT/'raw'/name
    receipt=json.loads((folder/'dispatcher_result.json').read_text());assert not receipt['complete']
    exits=json.loads((folder/'worker_exits.json').read_text());assert not exits['passed']
    hold=ROOT/'state/D_kernelonly_profiles_hold.json';assert hold.exists()
    write(ROOT/'analysis/D_kernelonly_profiler_failure_review.json',dict(time=time.time(),name=name,
        evidence='FAILED',dispatcher=receipt,worker_exits=exits,
        coordinator_completed_requests=200,trace_primary_eligible=False,
        identified_boundary='Decode worker failed inside CUPTI activity_end at collector close.',
        exact_collector_return_code='NOT_CAPTURED_IN_FAILED_PROCESS',
        root_cause='NOT_IDENTIFIED. The collector can report buffer exhaustion or a CUPTI record/drop-query error. The existing log does not distinguish these causes.',
        action='Preserve all raw files. Do not rerun this D trace to seek a passing timing. Leave Token clean-after NOT_RUN. Continue independent source/credit/cut research.',
        hold_SHA256=digest(hold)))
    write(ROOT/'state/acknowledged_optional_failures.json',dict(time=time.time(),holds=[hold.name],
        scope='Acknowledged failed optional profiler phase, not a resolved collector defect or a passing measurement.'))
    write(ROOT/'state/D_kernelonly_profiles_complete.json',dict(time=time.time(),complete=True,status='FAILED',
        remaining_cells=[dict(axis='Token',mode='clean_after',status='NOT_RUN',reason='Profile worker close failed.')]))


if __name__=='__main__':main()
