"""Measure host boundary identities without relabeling them GPU compute."""
import collections
import json
import statistics
from .common import ROOT,write
from ..dense_ar_slo_layer_pd.analyze import percentile


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summary(values):
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
        p50=percentile(values,.5),p95=percentile(values,.95),maximum=max(values,default=None))


def analyze(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    if cfg['arm']!='PD_G1':
        return None
    source={r['request_id']:r['actual_send_start'] for r in records(folder/'actual_submissions.jsonl')}
    destination={r['request_id']:r['actual_send_start'] for r in records(folder/'D_actual_submissions.jsonl')}
    ready={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    paths=[]
    for rid,row in requests.items():
        points=[row['arrival'],source[rid],ready[rid],destination[rid],row['token_events'][0]['worker_time'],row['first_token']]
        spans={key:1000*(b-a) for key,a,b in zip(
            ('arrival_to_P_host_submit','P_submit_to_source_notification',
             'source_notification_to_D_host_submit','D_submit_to_first_worker_notice',
             'first_worker_notice_to_client'),points,points[1:])}
        assert abs(sum(spans.values())-row['ttft_ms'])<1e-5
        paths.append(dict(request_id=rid,TTFT_ms=row['ttft_ms'],boundaries_monotonic=points,
            spans_ms=spans,ordered=all(b>=a for a,b in zip(points,points[1:])),
            first_event_source=row['token_events'][0].get('source')))
    groups=collections.defaultdict(list)
    for row in records(folder/'D_host_admission.jsonl'):
        groups[row['cohort']].append(row)
    cohorts=[]
    for number,members in groups.items():
        rows=[requests[r['request_id']] for r in members]
        first=min(r['token_events'][0]['worker_time'] for r in rows)
        finish=max(r['token_events'][-1]['worker_time'] for r in rows)
        client_finish=max(r['finish'] for r in rows)
        host_submit=min(destination[r['request_id']] for r in members)
        worker_intervals=[1000*(b['worker_time']-a['worker_time']) for row in rows
                          for a,b in zip(row['token_events'],row['token_events'][1:])]
        cohorts.append(dict(cohort=number,requests=[r['request_id'] for r in members],size=len(rows),
            D_host_submit=host_submit,first_worker_notice=first,last_worker_notice=finish,
            last_client_notice=client_finish,preparation_host_ms=1000*(first-host_submit),
            worker_decode_residency_ms=1000*(finish-first),
            D_host_admission_to_last_worker_ms=1000*(finish-host_submit),
            worker_decode_residency_per_request_ms=1000*(finish-first)/len(rows),
            mean_worker_ITL_ms=statistics.mean(worker_intervals)))
    cohorts.sort(key=lambda r:r['D_host_submit'])
    for previous,current in zip(cohorts,cohorts[1:]):
        current['host_gap_after_prior_worker_finish_ms']=1000*(current['D_host_submit']-previous['last_worker_notice'])
        current['host_gap_after_prior_client_finish_ms']=1000*(current['D_host_submit']-previous['last_client_notice'])
    value=dict(name=name,method=cfg['evaluation_arm'],layers=cfg['static_layers'],cap=cfg['cap'],
        QPS=cfg['arrival_rate'],seed=cfg['arrival_seed'],count=cfg['count'],admission=cfg['D_admission_batch'],
        request_paths=paths,cohorts=cohorts,
        summaries={key:summary([r['spans_ms'][key] for r in paths]) for key in paths[0]['spans_ms']},
        cohort_sizes=dict(collections.Counter(r['size'] for r in cohorts)),
        residency_ms=summary([r['worker_decode_residency_ms'] for r in cohorts]),
        residency_per_request_ms=summary([r['worker_decode_residency_per_request_ms'] for r in cohorts]),
        summed_cohort_decode_residency_per_completed_request_ms=sum(r['worker_decode_residency_ms'] for r in cohorts)/len(paths),
        summed_cohort_preparation_per_completed_request_ms=sum(r['preparation_host_ms'] for r in cohorts)/len(paths),
        preparation_ms=summary([r['preparation_host_ms'] for r in cohorts]),
        physical_kernel_time='NOT_MEASURED',SM_HBM_traffic='NOT_MEASURED',
        evidence='TRACE_ANALYSIS',
        scope='Exact host-clock TTFT identity. Source enclosure includes computation, export and host work. D preparation includes transport, import, compute and scheduling. Cohort residency is a worker-notice enclosure, not exclusive GPU busy. Native first-token notice follows the existing source-token release contract. Mean stage differences can be added as host identities. Differences of stage p95 are not a decomposition of TTFT p95.')
    write(ROOT/'analysis'/(name+'_host_paths.json'),value)
    return value


if __name__=='__main__':
    import sys
    for name in sys.argv[1:]:
        analyze(name)
