"""Separate accumulated source service from credit and host admission gaps."""
import json
import numpy as np
from .common import ROOT,PREVIOUS,write
from .queue_accounting import lines


def analyze(name,root):
    folder=root/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    credit=cfg['source_credit_limit'];assert cfg.get('controlled_pipeline') and not cfg.get('strict_serial_pipeline')
    requests=sorted(lines(folder/'requests.jsonl'),key=lambda r:r['sample_id'])
    decisions={r['request_id']:r['time'] for r in lines(folder/'axis_decisions.jsonl')}
    handoffs={r['request_id']:r for r in lines(folder/'handoffs.jsonl')}
    ready={rid:r['received'] for rid,r in handoffs.items()}
    queries=json.loads((folder/'actual_queries_prefill.json').read_text())['rows']
    rows=[]
    for i,r in enumerate(requests):
        rid=f'q{r["sample_id"]}';start=decisions[rid];previous=ready[f'q{requests[i-1]["sample_id"]}'] if i else r['arrival']
        release=requests[i-credit]['first_token'] if i>=credit else r['arrival']
        first_prepare=min(q['time'] for q in queries if q['request_id']==rid)
        eligible=max(r['arrival'],previous,release)
        assert start>=eligible-1e-6
        rows.append(dict(arrival=r['arrival'],P_start=start,P_ready=ready[rid],
            P_service=ready[rid]-start,arrival_gap=max(0.,r['arrival']-previous),
            credit_extension=max(0.,release-max(r['arrival'],previous)),
            eligible_host_gap=max(0.,start-eligible),prepare_offset=first_prepare-start,
            prepare_to_worker_ready=handoffs[rid]['source_ready']-first_prepare,
            worker_to_parent_ready=ready[rid]-handoffs[rid]['source_ready'],
            observed_P_queue=first_prepare-r['arrival']))
    n=len(rows);origin=rows[0]['arrival'];weighted=lambda key:1000*sum((n-1-j)/n*rows[j][key] for j in range(n-1))
    later=lambda key:1000*sum((n-j)/n*rows[j][key] for j in range(1,n))
    parts=dict(initial_P_dispatch=1000*(rows[0]['P_start']-origin),
        weighted_prior_source_host_service=weighted('P_service'),
        weighted_arrival_gaps=later('arrival_gap'),weighted_source_credit_extension=later('credit_extension'),
        weighted_eligible_host_admission_gaps=later('eligible_host_gap'),
        mean_current_dispatch_to_prepare=1000*float(np.mean([r['prepare_offset'] for r in rows])),
        negative_mean_arrival_offset=-1000*float(np.mean([r['arrival']-origin for r in rows])))
    observed=1000*float(np.mean([r['observed_P_queue'] for r in rows]))
    assert abs(sum(parts.values())-observed)<1e-3,(name,parts,observed)
    return dict(name=name,N=n,credit=credit,mean_P_queue_ms=observed,exact_parts_ms=parts,
        weighted_prior_source_service_subparts_ms={key:weighted(key) for key in ('prepare_offset','prepare_to_worker_ready','worker_to_parent_ready')},
        source_service_mean_ms=1000*float(np.mean([r['P_service'] for r in rows])),
        evidence='TRACE_ANALYSIS',scope='Exact source-chain accounting on recorded request IDs. Source host service includes GPU work, IPC, preparation and notification. '
        'Credit changes can also affect measured service. Weighted terms are not independently randomized causal effects or exclusive GPU costs.')


def main():
    historical={axis:analyze(f'ControlledDiscovery_{axis}4_N200_q2.35_s2026112061_BI0_v3',PREVIOUS)
        for axis in ('Layer','Token')}
    difference={key:historical['Layer']['exact_parts_ms'][key]-historical['Token']['exact_parts_ms'][key]
        for key in historical['Layer']['exact_parts_ms']}
    write(ROOT/'analysis/exact_source_chain_decomposition_v1.json',dict(historical_Last4=historical,
        Layer_minus_Token_exact_queue_parts_ms=difference,
        observed_P_queue_difference_ms=historical['Layer']['mean_P_queue_ms']-historical['Token']['mean_P_queue_ms'],
        Layer_minus_Token_weighted_prior_source_service_subparts_ms={key:
            historical['Layer']['weighted_prior_source_service_subparts_ms'][key]-historical['Token']['weighted_prior_source_service_subparts_ms'][key]
            for key in historical['Layer']['weighted_prior_source_service_subparts_ms']},
        scope='Same original Natural200 IDs, arrival seed2061,cut4 and c4. Never add cut8 serial costs to this identity. No GPU attribution from an unqualified Natural profile.'))
    print(json.dumps(difference,indent=2))


if __name__=='__main__':main()
