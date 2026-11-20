"""Keep clean P activity, D host quiescence and safe commits separate."""
import json
from .common import ROOT,write
from .serving_analysis import stats,time_weighted
from .profile_results import union


def main():
    results=[]
    for family in ('ContinuousNative','ControlledNative'):
        source=ROOT/'analysis'/f'BI0_P_only_{family}_observer_v2.json'
        if not source.exists():continue
        audit=json.loads(source.read_text());name=audit['names']['profile'];folder=ROOT/'raw'/name
        left,right=audit['kernel_activity']['common_host_window']
        queues=json.loads((folder/'scheduler_queues_decode.json').read_text())['rows']
        witnesses=json.loads((folder/'D_quiescent_witness.json').read_text())['rows']
        spans=[(max(left,r['start']),min(right,r['end'])) for r in witnesses if r['start']<right and r['end']>left]
        hands=[json.loads(line) for line in (folder/'handoffs.jsonl').read_text().splitlines()]
        hands=[r for r in hands if left<=r['source_ready']<=right]
        aligned=sum(any(a<=r['source_ready']<=b for a,b in spans) for r in hands)
        cert_file=folder/'idle_certificates.jsonl'
        certs=[json.loads(line) for line in cert_file.read_text().splitlines()] if cert_file.exists() else []
        results.append(dict(family=family,name=name,observer_gate_passed=audit['observer_gate_passed'],
            P_kernel_busy_fraction=audit['kernel_activity']['roles']['prefill']['kernel_busy_fraction'],
            P_queue=audit['kernel_activity']['roles']['prefill']['sampled_queue_state_same_window']['waiting_total'],
            D_queues={key:time_weighted(queues,key,dict(start=left,end=right)) for key in
                ('waiting_total','remote_KV_waiting','decode_ready_waiting','active_decode')},
            D_stream_completed_scheduler_empty=dict(intervals=len(spans),union_seconds=union(spans),
                observed_fraction=union(spans)/(right-left),length_ms=stats(1000*(b-a) for a,b in spans),
                scope='Completed current-stream event and empty scheduler/engine. Not a D kernel timeline or a predicted future budget.'),
            full_native_P_ready_alignment=dict(source_ready_requests=len(hands),inside_observed_empty_interval=aligned,
                scope='Full-Native ready times. A hypothetical earlier layer/token cut changes readiness and future queues. No frozen-state serving-gain claim.'),
            device_wide_empty_admission_certificates=dict(n=len(certs),all_passed=bool(certs) and all(r['passed'] for r in certs)),
            D_clean_kernel_busy='NOT_MEASURED_IN_P_ONLY_INTERVENTION',
            evidence='MEASURED_P_KERNEL_ACTIVITY_AND_D_HOST_STATE' if audit['observer_gate_passed'] else 'DIAGNOSTIC_OBSERVER_GATE_FAILED'))
    write(ROOT/'analysis/BI0_prefill_heavy_opportunity_v2.json',dict(results=results,
        scope='Same host window for P kernels and D sampled state. Cross-device kernels are not combined. '
        'Host quiescence does not certify every physical idle gap. Device-wide empty admission protects existing decode by serialization, '
        'not by preemption or active-decode deadline prediction. No sustainable-capacity estimate.'))
    return results


if __name__=='__main__':main()
