"""Screen a separate execution idea before implementing scheduler state."""
import statistics,collections
from .common import *
from .analyze import records,percentile


def main():
    calibration='canonical_cost_B8_B_N120_v1'
    rows=json.loads((ROOT/'analysis'/(calibration+'_calibration.json')).read_text())['rows']
    rows=[r for r in rows if r['layers']==12]
    reference='async_D_STATIC_L12_N128_O16_q4.8_v1';folder=ROOT/'raw'/reference
    requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')};groups={}
    for row in records(folder/'D_host_admission.jsonl'):groups.setdefault(row['cohort'],[]).append(row['request_id'])
    decode=[1000*(max(requests[r]['finish'] for r in ids)-min(requests[r]['first_token'] for r in ids))
        for ids in groups.values() if len(ids)==4]
    prefix=statistics.mean(r['P_prefix_host_ms'] for r in rows)
    components={k:statistics.mean(r[k] for r in rows) for k in
        ('handoff_to_D_stream_start_ms','D_suffix_event_ms','D_commit_host_ms')}
    destination=sum(components.values())+2*statistics.mean(decode)
    write(ROOT/'analysis/decode_microcohort_decoupling_screen.json',dict(time=time.time(),
        hypothesis='Execute one legal D8 suffix, then serve two D4 Decode cohorts. The second cohort would need delayed first-output admission and retained KV.',
        source_calibration=calibration,Decode_reference=reference,source_samples=len(rows),Decode_cohort_samples=len(decode),
        P_prefix_Host_mean_ms=prefix,component_means_ms=components,
        D4_first_to_last_delivery_mean_ms=statistics.mean(decode),
        D4_first_to_last_delivery_p95_ms=percentile(decode,.95),
        modeled_D8_service_with_two_D4_cohorts_ms=destination,
        modeled_bottleneck_requests_s=8000/max(prefix,destination),
        additional_control_overhead_included=False,actual_execution='NOT_RUN',
        decision='Do not implement the extra parked-request scheduler now. The component screen does not show sufficient margin over the strong Colocated frontier. Keep the ordinary qualified D4 and D8 serving comparisons.',
        evidence='STRUCTURAL_ACCOUNTING',estimate_kind='MODELED_COMPONENT_COST_SCREEN',
        scope='Combine isolated enclosures with a different observed discovery run. This is neither an implemented serving method nor a physical upper bound. Components can change after integration. No Decode safety or sustainable speedup is proved. A future prototype requires exact paused-request lifecycle and honest first-output timing.'))
    print('DECODE_MICROCOHORT_MODELED_SCREEN',8000/max(prefix,destination),flush=True)


if __name__=='__main__':main()
