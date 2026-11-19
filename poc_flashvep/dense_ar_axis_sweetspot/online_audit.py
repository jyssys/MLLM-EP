"""Separate exact client boundary identities from physical GPU measurements."""
import collections
import json
import statistics
from .common import ROOT,bind_legacy,write,digest
from .historical_audit import classify,records


def audit(name):
    bind_legacy()
    from ..dense_ar_matched_axis_pd.analysis import metrics
    from ..dense_ar_matched_axis_pd.host_paths import analyze
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    value=metrics(name)
    if cfg.get('matched_D_cohort_size',1)>2:
        from .cohort_audit import audit as geometry
    elif cfg.get('matched_D_cohort_size'):
        from ..dense_ar_matched_axis_pd.fixed_cohort_audit import audit as geometry
    else:
        from ..dense_ar_matched_axis_pd.batch_matched import audit as geometry
    geometry(name)
    host=analyze(name)
    requests={r['request_id']:r for r in host['request_paths']}
    cohorts=[]
    for c in host['cohorts']:
        times=[requests[r]['boundaries_monotonic'][2] for r in c['requests']]
        cohorts.append(dict(first_ready=min(times),all_ready=max(times),start=c['D_host_submit'],
            finish=c['last_client_notice'],requests=c['requests']))
    pieces=[]
    for index,c in enumerate(cohorts):
        for rid in c['requests']:
            r=requests[rid]
            ready,submitted=r['boundaries_monotonic'][2:4]
            pieces.append(dict(request_id=rid,components_ms=classify(ready,submitted,c['all_ready'],cohorts[:index])))
    folder=ROOT/'raw'/name
    reservations=records(folder/'reservation_observations.jsonl')
    common_retries=records(folder/'common_credit_retries.jsonl')
    duration=sum(c['last_client_notice']-c['D_host_submit'] for c in host['cohorts'])
    result=dict(name=name,method=cfg['evaluation_arm'],layers=cfg['static_layers'],
        output=cfg['cap'],QPS=cfg['arrival_rate'],seed=cfg['arrival_seed'],count=cfg['count'],
        actual_batch=cfg['D_admission_batch'],source_window=cfg['source_outstanding_limit'],
        P_submit_to_ready_mean_ms=host['summaries']['P_submit_to_source_notification']['mean'],
        D_host_slot_per_request_ms=1000*duration/cfg['count'],
        Ready_D_mean_ms=host['summaries']['source_notification_to_D_host_submit']['mean'],
        P_external_wait_mean_ms=host['summaries']['arrival_to_P_host_submit']['mean'],
        wait_components_mean_ms={k:statistics.mean(r['components_ms'].get(k,0.) for r in pieces)
            for k in {k for r in pieces for k in r['components_ms']}},
        reservation_rejections=sum(not r['accepted'] for r in reservations),
        common_credit_retries=len(common_retries),
        request_components=pieces,
        exact_identity_passed=all(r['ordered'] for r in host['request_paths']),
        metric_summary={k:value[k] for k in ('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms',
            'ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms','output_tokens_s','completed_QPS')},
        common_Balanced=value['SLO_regimes']['Common_Balanced_OutputSweep'],
        host_paths_SHA256=digest(ROOT/'analysis'/(name+'_host_paths.json')),
        evidence='MEASURED_ONLINE',decomposition_evidence='TRACE_ANALYSIS',
        physical_GPU_service='NOT_MEASURED',
        scope='Finite-cohort client metrics and disjoint host guards. Host service boundaries include scheduling and transport. They are not exclusive GPU compute or physical idle.')
    assert result['exact_identity_passed']
    write(ROOT/'analysis'/(name+'_service_queue_audit.json'),result)
    return result
