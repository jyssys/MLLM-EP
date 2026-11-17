"""Compare online predictions with observed group completion boundaries."""
import collections
import json
import statistics
from .common import ROOT,write
from .analysis import records
from ..dense_ar_slo_layer_pd.analyze import percentile


def summarize(values):
    return dict(count=len(values),mean=statistics.mean(values),p50=percentile(values,.5),
        p95=percentile(values,.95),p99=percentile(values,.99),maximum=max(values)) if values else dict(count=0)


def main():
    runs=[];samples=[]
    for path in sorted((ROOT/'raw').glob('*/online_decisions.jsonl')):
        folder=path.parent;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        cfg=json.loads((folder/'configuration.json').read_text())
        if not cfg.get('flow_balance_policy') or cfg.get('correctness_only') or cfg.get('execution_events'):
            continue
        commits=records(folder/'actual_cut_commits.jsonl')
        requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
        ready={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
        decisions=records(path);byrequests={tuple(r['requests']):r for r in commits}
        errorsP=[];errorsF=[];bounds=[];costs=[];rejects=collections.Counter();choices=collections.Counter()
        for index,row in enumerate(decisions):
            costs.append(row['policy_decision_host_ms'])
            for candidate in row['choice']['candidates']:
                rejects.update(candidate['rejection_reasons'])
            commit=commits[index];members=tuple(commit['requests']);m=commit['offloaded_layers'];choices[m]+=len(members)
            candidate=next((r for r in row['choice']['candidates'] if r['layers']==m),None)
            if candidate is None:
                continue
            actualP=max(ready[rid] for rid in members);actualF=max(requests[rid]['first_token'] for rid in members)
            ep=1000*(actualP-candidate['predicted_P_cut_ready'])
            ef=1000*(actualF-candidate['predicted_D_finish_mean'])
            eb=1000*(actualF-candidate['predicted_D_finish'])
            errorsP.append(ep);errorsF.append(ef);bounds.append(eb)
            samples.append(dict(name=folder.name,requests=list(members),layers=m,
                P_ready_error_ms=ep,first_output_error_ms=ef,upper_bound_error_ms=eb,
                own_TTFT_deadlines=row['observation'].get('own_TTFT_deadlines'),
                actual_first_outputs=[requests[rid]['first_token'] for rid in members],
                decision_time=row['time'],known_future_groups=row['observation'].get('visible_fifo_groups',[])))
        cohorts=collections.defaultdict(list)
        for row in records(folder/'D_host_admission.jsonl'):
            cohorts[row['cohort']].append(row)
        gaps=[];previous=None
        for group in sorted(cohorts.values(),key=lambda group:min(r['time'] for r in group)):
            if previous is not None:
                gaps.append(1000*(min(r['time'] for r in group)-max(requests[r['request_id']]['finish'] for r in previous)))
            previous=group
        runs.append(dict(name=folder.name,phase=cfg['phase'],choices=dict(choices),
            source_ready_MAE_ms=statistics.mean(abs(x) for x in errorsP) if errorsP else None,
            source_ready_absolute_error_ms=summarize([abs(x) for x in errorsP]),
            first_output_MAE_ms=statistics.mean(abs(x) for x in errorsF) if errorsF else None,
            first_output_absolute_error_ms=summarize([abs(x) for x in errorsF]),
            completion_underprediction_fraction=sum(x>0 for x in errorsF)/len(errorsF) if errorsF else None,
            upper_bound_coverage=sum(x<=0 for x in bounds)/len(bounds) if bounds else None,
            upper_bound_error_ms=summarize(bounds),controller_host_ms=summarize(costs),
            multi_gate_rejection_counts=dict(rejects),
            D_host_admission_before_old_cohort_client_retirement=sum(gap<-.01 for gap in gaps),
            host_retirement_to_next_admission_ms=summarize(gaps),
            evidence='TRACE_ANALYSIS',
            scope='Errors are actual minus predicted. Samples are correlated source groups, not independent starts or WCET evidence. Completion nominal uses stage medians and guarded availability; it is not an unbiased mean. Rejection counts overlap. Host retirement gaps are not physical kernel idle.'))
    write(ROOT/'analysis/online_prediction_and_deadline_validation.json',dict(runs=runs,samples=samples,
        evidence='TRACE_ANALYSIS',physical_safe_GPU_idle='NOT_MEASURED',
        scope='Read only recorded online observations, predictions and actual completions. No refitting uses held-out results. Known FIFO groups must already have arrived. Unknown future arrivals are not certified.'))
    print('PREDICTION_VALIDATION_STARTS',len(runs),'GROUP_OBSERVATIONS',len(samples),flush=True)


if __name__=='__main__':
    main()
