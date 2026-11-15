"""Audit realized batching and transport without assuming equal geometry."""
import collections
from .common import *
from .analyze import records,percentile


def structural_KV_bytes(tokens,layers):
    return tokens*layers*2*8*128*2


def audit(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    rows=records(folder/'requests.jsonl');byid={'q'+str(r['sample_id']):r for r in rows}
    cuts=collections.Counter(r.get('completed_D_prefill_layers',0) for r in rows)
    commits=records(folder/'actual_cut_commits.jsonl')
    source_macros=collections.Counter((len(r['requests']),r['offloaded_layers']) for r in commits)
    prompt_shapes=collections.Counter();stats=collections.defaultdict(list)
    for role in ('prefill','decode'):
        for path in folder.glob('native_'+role+'_*.json'):
            value=json.loads(path.read_text())
            if role=='prefill':
                prompt_shapes.update(r['scheduled_requests'] for r in value['prompt_spans']
                    if r['request_id'].split('-')[0] in byid)
            if role=='decode':
                for event in value['NIXL_stats_intervals']:
                    for key,values in event['data'].items():stats[key].extend(values)
    admissions=records(folder/'D_host_admission.jsonl');cohorts={}
    for row in admissions:cohorts.setdefault(row['cohort'],[]).append(row)
    cohort_shapes=collections.Counter(len(v) for v in cohorts.values());retirement_gaps=[]
    previous=None
    for group in sorted(cohorts.values(),key=lambda g:min(r['time'] for r in g)):
        if previous is not None:
            retirement_gaps.append(1000*(min(r['time'] for r in group)-
                max(byid[r['request_id']]['finish'] for r in previous)))
        previous=group
    host_retention=[];handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    for r in admissions:
        if r['request_id'] in handoffs:host_retention.append(1000*(r['time']-handoffs[r['request_id']]))
    def summary(values):
        return dict(samples=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
            maximum=max(values,default=None),minimum=min(values,default=None))
    # Each token has K and V, eight KV heads, 128 elements per head, and
    # two bytes per BF16 element. The result is4096 bytes per layer/token.
    expected_KV=sum(structural_KV_bytes(r['input_tokens'],36-r.get('completed_D_prefill_layers',0)) for r in rows)
    actual_KV=sum(stats['bytes_transferred'])
    structural_hidden=sum(r['input_tokens']*4096*2*2 for r in rows if r.get('completed_D_prefill_layers',0))
    return dict(name=name,requests=len(rows),actual_cut_histogram=dict(cuts),
        migrated_requests=sum(n for k,n in cuts.items() if k),completed_migrated_layer_work=sum(k*n for k,n in cuts.items()),
        total_prefill_layer_work=36*len(rows),
        source_macro_histogram=[dict(requests=b,layers=k,groups=n) for (b,k),n in sorted(source_macros.items())],
        source_prompt_request_observation_histogram=dict(prompt_shapes),
        destination_cohort_histogram=dict(cohort_shapes),
        prior_cohort_finish_to_next_host_admission_ms=summary(retirement_gaps),
        host_admission_before_prior_cohort_finish=sum(v<-.01 for v in retirement_gaps),
        source_ready_to_D_host_admission_ms=summary(host_retention),
        NIXL_reported_KV_bytes=actual_KV,structural_expected_partial_KV_bytes=expected_KV,
        NIXL_KV_byte_accounting_equal=actual_KV==expected_KV if stats['bytes_transferred'] else None,
        structural_Hidden_Residual_bytes=structural_hidden,
        NIXL_transfer_telemetry_ms=summary([1000*v for v in stats['transfer_duration']]),
        NIXL_post_telemetry_ms=summary([1000*v for v in stats['post_duration']]),
        scope='Actual completed cut and Host cohort geometry. Prompt shape counts weight request observations, not unique GPU launches. NIXL reports seconds and bytes in its native telemetry. Its operation durations can overlap and are not additive request critical paths. Hidden/Residual bytes are structural accounting. Host admission after prior cohort output supports the retirement guard, not physical GPU idle.',
        evidence='TRACE_ANALYSIS_AND_STRUCTURAL_ACCOUNTING')


def main():
    names=[p.parent.name for p in sorted((ROOT/'raw').glob('*/actual_cut_commits.jsonl'))
        if (p.parent/'dispatcher_result.json').exists() and
        json.loads((p.parent/'dispatcher_result.json').read_text()).get('complete')]
    values=[audit(name) for name in names]
    write(ROOT/'analysis/geometry_transport_audit_v2.json',dict(runs=values,analysis_version=2,
        prior_analysis_preserved='geometry_transport_audit.json',
        erratum='Version1 incorrectly used32768 rather than4096 bytes per layer/token in its expected-byte formula. Its expected KV totals were eight times too large. Actual NIXL telemetry, runtime payloads, state correctness, latency and online decisions are unchanged. Version2 uses2 K/V planes ×8 KV heads ×128 head elements ×2 BF16 bytes.'))
    print('GEOMETRY_TRANSPORT_RUNS',len(values),flush=True)


if __name__=='__main__':main()
