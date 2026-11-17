"""Count legal candidate decisions without treating host budgets as GPU idle."""
import collections
import json
import statistics
from .common import ROOT,write
from .analysis import records
from ..dense_ar_slo_layer_pd.analyze import percentile


def summary(values):
    if not values:
        return dict(count=0)
    mean=statistics.mean(values);sd=statistics.pstdev(values)
    return dict(count=len(values),mean=mean,standard_deviation=sd,
        coefficient_of_variation=sd/abs(mean) if mean else None,
        p10=percentile(values,.1),p50=percentile(values,.5),p90=percentile(values,.9),
        p95=percentile(values,.95),minimum=min(values),maximum=max(values))


def main():
    runs=[]
    for path in sorted((ROOT/'raw').glob('*/online_decisions.jsonl')):
        folder=path.parent;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        cfg=json.loads((folder/'configuration.json').read_text())
        if not cfg.get('flow_balance_policy') or cfg.get('correctness_only'):
            continue
        accepted=collections.Counter();chosen=collections.Counter();rejected=collections.Counter()
        residuals=[];available_gaps=[];pending=[];credits=[];different_best=0;P_and_safe=0
        rows=records(path)
        for row in rows:
            observation=row['observation'];choice=row['choice'];chosen[choice['layers']]+=1
            legal=[v['layers'] for v in choice['candidates'] if v['accepted']]
            accepted.update(legal)
            for candidate in choice['candidates']:
                rejected.update(candidate['rejection_reasons'])
            candidate=next((v for v in choice['candidates'] if v['layers']==choice['layers']),None)
            if candidate and observation.get('own_TTFT_deadlines'):
                residuals.append(1000*(min(observation['own_TTFT_deadlines'])-candidate['predicted_D_finish']))
                available_gaps.append(1000*max(0.,candidate['predicted_P_cut_ready']-observation['D_available']))
            pending.append(observation['pending_requests']);credits.append(observation['KV_free_blocks'])
            different_best+=int(choice['layers']!=max(legal)) if legal else 0
            P_and_safe+=int(observation['P_backlog']>0 and any(m>0 for m in legal))
        runs.append(dict(name=folder.name,phase=cfg.get('phase'),decisions=len(rows),
            feasible_candidate_decisions=dict(accepted),chosen_source_groups=dict(chosen),
            multi_gate_rejection_counts=dict(rejected),
            selected_nonlargest_feasible_cut=different_best,
            P_backlog_and_accepted_migration_decisions=P_and_safe,
            predicted_first_output_deadline_residual_ms=summary(residuals),
            nominal_D_available_to_P_cut_gap_ms=summary(available_gaps),
            D_pending_requests=summary(pending),D_KV_free_blocks=summary(credits),
            evidence='TRACE_ANALYSIS',
            scope='Counts use actual recorded online decisions. Candidate rejections overlap. The gap and residual are model outputs, not physical GPU idle, not a suffix execution guarantee and not independent-start statistics. A chosen nonlargest cut can reflect cost/queue scoring, not only deadline rejection.'))
    write(ROOT/'analysis/online_safe_budget_variance.json',dict(runs=runs,
        physical_request_feasible_GPU_slack='NOT_MEASURED',
        scope='Host-state opportunity and model uncertainty remain separate from measured serving benefit. Inputs other than8000 conservatively choose Native in this policy.'))
    print('BUDGET_VARIANCE_STARTS',len(runs),flush=True)


if __name__=='__main__':
    main()
