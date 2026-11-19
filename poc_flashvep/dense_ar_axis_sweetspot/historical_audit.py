"""Reconstruct measured host guards without calling them GPU activity."""
import collections
import json
import statistics
import time
from .common import PREVIOUS, ROOT, digest, initialize, write


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def classify(ready, submitted, partner_ready, prior):
    assert submitted>=ready
    boundaries={ready,submitted}
    for value in [partner_ready]+[x for c in prior for x in (c['first_ready'],c['all_ready'],c['start'],c['finish'])]:
        if ready<value<submitted:
            boundaries.add(value)
    output=collections.Counter()
    points=sorted(boundaries)
    for left,right in zip(points,points[1:]):
        middle=(left+right)/2
        active=any(c['start']<=middle<c['finish'] for c in prior)
        own=middle<partner_ready
        if active:
            reason='prior_cohort_guard_and_own_partner' if own else 'prior_cohort_guard_only'
        elif own:
            reason='own_partner_only'
        elif any(c['first_ready']<=middle<c['all_ready'] for c in prior):
            reason='earlier_FIFO_cohort_partner_only'
        elif any(c['all_ready']<=middle<c['start'] for c in prior):
            reason='earlier_FIFO_eligible_collection_or_host_gap'
        else:
            reason='eligibility_residual_NOT_INSTRUMENTED'
        output[reason]+=1000*(right-left)
    assert abs(sum(output.values())-1000*(submitted-ready))<1e-5
    return dict(output)


def analyze(start):
    name=start['name'];folder=PREVIOUS/'raw'/name
    path=PREVIOUS/'analysis'/(name+'_host_paths.json')
    host=json.loads(path.read_text())
    request={r['request_id']:r for r in host['request_paths']}
    cohorts=[]
    for c in host['cohorts']:
        times=[request[r]['boundaries_monotonic'][2] for r in c['requests']]
        cohorts.append(dict(cohort=c['cohort'],requests=c['requests'],first_ready=min(times),
            all_ready=max(times),start=c['D_host_submit'],finish=c['last_client_notice']))
    cohorts.sort(key=lambda c:c['start'])
    rows=[]
    for i,c in enumerate(cohorts):
        for rid in c['requests']:
            r=request[rid];ready=r['boundaries_monotonic'][2];submitted=r['boundaries_monotonic'][3]
            pieces=classify(ready,submitted,c['all_ready'],cohorts[:i])
            rows.append(dict(request_id=rid,ready=ready,D_submit=submitted,
                Ready_to_D_ms=1000*(submitted-ready),components_ms=pieces))
    components={k:statistics.mean(r['components_ms'].get(k,0.) for r in rows)
                for k in sorted({k for r in rows for k in r['components_ms']})}
    reservations=records(folder/'reservation_observations.jsonl')
    retries=records(folder/'common_credit_retries.jsonl')
    runtime=records(folder/'runtime_states.jsonl')
    p=host['summaries']['P_submit_to_source_notification']['mean']
    d=sum(1000*(c['finish']-c['start']) for c in cohorts)/start['count']
    queued_mean=statistics.mean(r['Ready_to_D_ms'] for r in rows)
    assert abs(sum(components.values())-queued_mean)<1e-5
    free=[r['KV_free_blocks']-r.get('reserved_future_blocks',0) for r in runtime if r['role']=='decode']
    first=[c['start'] for c in cohorts];last=[c['finish'] for c in cohorts]
    return dict(name=name,stage=start['stage'],method=start['method'],layers=start['layers'],
        output=start['cap'],QPS=start['QPS'],seed=start['seed'],count=start['count'],
        actual_cohort_histogram=host['cohort_sizes'],mean_Ready_to_D_ms=queued_mean,
        mean_wait_components_ms=components,requests=rows,
        P_submit_to_notification_mean_ms=p,D_cohort_host_occupancy_per_request_ms=d,
        effective_stage_balance_D_over_P=d/p,
        modeled_saturated_cycle_ms=max(p,d),modeled_unqueued_path_ms=p+host['summaries']['D_submit_to_first_worker_notice']['mean'],
        D_cohort_guard_span_ms=1000*(max(last)-min(first)),
        prior_guard_overlap_ms_per_request=sum(components.get(k,0) for k in
            ('prior_cohort_guard_and_own_partner','prior_cohort_guard_only')),
        reservation_count=len(reservations),reservation_rejections=sum(not r['accepted'] for r in reservations),
        reservation_negotiation_mean_ms=statistics.mean(r['negotiation_ms'] for r in reservations) if reservations else None,
        common_credit_retries=len(retries),minimum_observed_D_legal_credit= min(free) if free else None,
        provenance=dict(host_paths_SHA256=digest(path),configuration_SHA256=digest(folder/'configuration.json'),
            reservations_SHA256=digest(folder/'reservation_observations.jsonl') if reservations else None),
        not_instrumented=['Exclusive GPU Forward','Exclusive GEMM/Attention','DMA start/end','Continuous admission-gate reject reasons'],
        evidence='TRACE_ANALYSIS',
        scope='Disjoint host-clock overlap accounting. Prior cohort guard spans include transfer, preparation, decode and delivery. They are not kernel busy or GPU idle. Residual time is not assigned to KV or DMA without instrumentation. Effective service demand includes host/transport overhead. A max(P,D) cycle is a model, not measured sustainable capacity.')


def main():
    initialize()
    data=json.loads((PREVIOUS/'analysis/independent_start_statistics.json').read_text())
    selected=[r for r in data['starts'] if r['stage'] in
              ('batch1_discovery','batch1_confirmation','cohort2_discovery','cohort2_confirmation')]
    results=[analyze(r) for r in selected]
    groups=collections.defaultdict(list)
    for row in results:
        groups[row['stage'],row['method'],row['layers'],row['output'],row['QPS']].append(row)
    aggregate=[]
    fields=('mean_Ready_to_D_ms','P_submit_to_notification_mean_ms','D_cohort_host_occupancy_per_request_ms',
            'effective_stage_balance_D_over_P','prior_guard_overlap_ms_per_request','modeled_saturated_cycle_ms',
            'modeled_unqueued_path_ms')
    for (stage,method,m,output,qps),values in sorted(groups.items()):
        aggregate.append(dict(stage=stage,method=method,layers=m,output=output,QPS=qps,
            starts=len(values),seeds=[r['seed'] for r in values],
            metrics={f:statistics.mean(r[f] for r in values) for f in fields},
            mean_wait_components_ms={k:statistics.mean(r['mean_wait_components_ms'].get(k,0.) for r in values)
                for k in sorted({k for r in values for k in r['mean_wait_components_ms']})},
            reservation_rejections=sum(r['reservation_rejections'] for r in values),
            credit_retries=sum(r['common_credit_retries'] for r in values)))
    write(ROOT/'analysis/historical_guard_decomposition.json',dict(time=time.time(),
        source_commit='4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7',starts=results,aggregate=aggregate,
        evidence='TRACE_ANALYSIS',scope='Historical observations only. These are not new GPU runs, exclusive GPU service times or causal counterfactuals. Separate discovery and confirmation.'))
    for r in aggregate:
        if r['stage']=='cohort2_confirmation' and r['method']!='Native':
            print(r['method'],r['layers'],{k:round(v,3) for k,v in r['metrics'].items()},
                  'rejects',r['reservation_rejections'],'retries',r['credit_retries'],flush=True)
    print('HISTORICAL_GUARD_AUDIT',len(results),flush=True)


if __name__=='__main__':
    main()
