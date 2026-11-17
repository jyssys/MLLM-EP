"""Replay the observed credit predicate, then test one bounded configuration."""
import collections
import json
from .common import ROOT,write
from .analysis import records
from .discovery import job


def main():
    source='flow_low_Static_fraction_P4_adaptiveD8_N128_v1';folder=ROOT/'raw'/source
    cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
    states=records(folder/'runtime_states.jsonl')
    latest={role:next(r for r in reversed(states) if r['role']==role) for role in ('prefill','decode')}
    admissions=records(folder/'D_host_admission.jsonl');cohort=max(r['cohort'] for r in admissions)
    members=[r['request_id'] for r in admissions if r['cohort']==cohort]
    extension=records(next(folder.glob('extension_decode_*.jsonl')))
    reads=[r['external_request_id'] for r in extension if r.get('event')=='NATIVE_REUSE_EXACT_PREFIX_READ' and r['external_request_id'] in members]
    ready=[r for r in extension if r.get('event')=='NATIVE_KV_COHORT_READY' and r.get('cohort')==cohort]
    D=latest['decode'];free=D['KV_free_blocks'];reserved=D['reserved_future_blocks']
    need=(cfg['length']+1+15)//16
    assert free-reserved<need
    assert len(set(reads))<len(members) and not ready
    result=dict(evidence='TRACE_ANALYSIS',source=source,latest_states=latest,
        last_cohort=cohort,members=members,submitted_prefix_reads=sorted(set(reads)),
        missing_prefix_reads=sorted(set(members)-set(reads)),cohort_ready_events=ready,
        free_blocks=free,reserved_future_blocks=reserved,effective_free_blocks=free-reserved,
        minimum_native_prompt_allocation_blocks=need,
        actual_allocator_predicate_rejects=free<reserved+need,
        source_free_blocks=latest['prefill']['KV_free_blocks'],source_prompt_blocks=cfg['length']//16,
        mechanism='Future committed suffix reservations can block unreserved Native members of an all-or-none cohort. The cohort cannot retire until every member imports. Future suffix work cannot be admitted before that cohort retires. Source retention can then fill the P KV pool.',
        scope='Read actual queue, reservation and submitted-read records and the unchanged allocator/barrier predicates. The failed process did not flush its final per-request allocation-refusal log. Exact callback ordering remains NOT_IDENTIFIED. This is an implementation credit-ordering diagnosis, not proof of GPU compute exhaustion.')
    write(ROOT/'analysis/mixed_cohort_credit_stall_trace.json',result)
    cfg.update(name='flow_low_Static_fraction_P4_adaptiveD8_credit16_diagnostic_v1',
        max_reserved_requests=16,phase='RUNTIME_CREDIT_STALL_DIAGNOSTIC',
        evaluation_arm='RuntimeDiagnostic',correctness_only=True,clean_timing=False,
        runtime_observation=True)
    value=job(cfg)
    write(ROOT/'analysis/mixed_cohort_credit16_diagnostic.json',dict(source=source,
        diagnostic=cfg['name'],complete=bool(value),metrics=value,
        evidence='CORRECTNESS_ONLY' if value else 'FAILED',
        scope='Change only the future reservation cap to16 in the same failed workload and enable diagnostic scheduler logging. Actual cut fallbacks can change. This is not a fair policy performance comparison or a new main runtime. No existing reservation or live state is revoked.'))


if __name__=='__main__':
    main()
