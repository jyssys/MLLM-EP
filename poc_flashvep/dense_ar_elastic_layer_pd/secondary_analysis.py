"""Compare paired controls without converting aggregate matching into a policy."""
from collections import Counter
from .common import *
from .analyze import records
from .statistics import paired_ratio
from .result_analysis import estimate


def main():
    primary=json.loads((ROOT/'state/confirmation_progress.json').read_text())['results']
    secondary=json.loads((ROOT/'state/secondary_progress.json').read_text())['results']
    primary=[r for r in primary if r['cap']==16 and r['rate']==4.8 and r['arm']!='Reference']
    rows=primary+secondary
    keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','maximum_ITL_ms','E2E_p95_ms','output_tokens_s',
        'P_source_production_span_s','P_clearance_after_last_arrival_s')
    result={}
    geometry=[]
    for row in secondary:
        if row['arm']!='MatchedProfile':continue
        dynamic=next(r for r in primary if r['arm']=='Revised' and r['seed']==row['seed'])
        def histogram(name):
            return Counter((r['offloaded_layers'],len(r['requests']))
                for r in records(ROOT/'raw'/name/'actual_cut_commits.jsonl'))
        actual=histogram(row['name']);reference=histogram(dynamic['name'])
        geometry.append(dict(seed=row['seed'],exact_request_and_layer_budget=row['realized_budget_match'],
            exact_realized_group_geometry=actual==reference,
            Dynamic_groups={f'Last{m}_B{b}':n for (m,b),n in reference.items()},
            Matched_groups={f'Last{m}_B{b}':n for (m,b),n in actual.items()}))
    for arm in sorted({r['arm'] for r in rows}):
        selected=[r for r in rows if r['arm']==arm]
        value=dict(names=[r['name'] for r in selected],independent_starts=len(selected),
            estimates={k:estimate([r['metrics'][k] for r in selected if r['metrics'].get(k) is not None]) for k in keys},
            SLO={label:{k:estimate([r['metrics']['SLO_regimes'][label][k] for r in selected])
                for k in ('Decode_attainment','Joint_attainment','Request_SLO_goodput')}
                for label in ('TTFT_priority','Balanced','Decode_priority')},ratios={})
        for reference in ('Native','Static','Colocated','Revised','MatchedProfile','ControllerZero'):
            baseline={r['seed']:r for r in rows if r['arm']==reference}
            paired=[r for r in selected if r['seed'] in baseline]
            if len(paired)<2:continue
            ratios={k:paired_ratio([r['metrics'][k] for r in paired],
                [baseline[r['seed']]['metrics'][k] for r in paired]) for k in keys
                if all(r['metrics'].get(k) is not None and baseline[r['seed']]['metrics'].get(k) is not None for r in paired)
                and sum(baseline[r['seed']]['metrics'][k] for r in paired)!=0}
            for label in ('TTFT_priority','Balanced','Decode_priority'):
                denominator=[baseline[r['seed']]['metrics']['SLO_regimes'][label]['Request_SLO_goodput'] for r in paired]
                if sum(denominator):ratios[label+'_goodput']=paired_ratio(
                    [r['metrics']['SLO_regimes'][label]['Request_SLO_goodput'] for r in paired],denominator)
            value['ratios'][reference]=ratios
        value['migration']=[dict(name=r['name'],seed=r['seed'],migrated_requests=r.get('migrated_requests'),
            migrated_layers=r.get('migrated_layers'),completed_layer_distribution=r.get('completed_layer_distribution'),
            realized_budget_match=r.get('realized_budget_match'),expected_requests=r.get('expected_matched_requests'),
            expected_layers=r.get('expected_matched_layers')) for r in selected]
        result[arm]=value
    write(ROOT/'analysis/matched_migration_and_ablation_results.json',dict(arms=result,geometry_audit=geometry,
        evidence='TRACE_ANALYSIS',cap=16,offered_QPS=4.8,
        scope='Three independent start pairs. MatchedProfile shuffles cut assignments derived from realized groups. Actual request/layer totals match, but arrivals and collection change the realized group geometry. It is post-hoc budget accounting, not a deployable held-out winner.'))
    write(ROOT/'analysis/matched_control_geometry_erratum.json',dict(results=geometry,
        evidence='TRACE_ANALYSIS',
        scope='The intended group-profile shuffle does not preserve actual source batches. Migration counts match, but the observed advantage includes batching and timing. Do not identify a pure predictor contribution from this comparison.'))
    print('SECONDARY_PAIRED_ARMS',len(result))


if __name__=='__main__':main()
