"""Verify actual work and geometry before computing paired matched ratios."""
import collections
import json
from .common import ROOT,write
from .analysis import metrics
from .statistics_report import ratio_interval


def main():
    pairs=[]
    keys=('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms',
        'output_tokens_s','completed_QPS')
    for seed in (2026111760,2026111761,2026111762):
        names={method:f'matched_confirm_{method}_Last12_N144_q6.4_s{seed}_v1'
            for method in ('Native','Static','Token')}
        if not all((ROOT/'raw'/name/'dispatcher_result.json').exists() and
            json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text()).get('complete')
            for name in names.values()):
            continue
        values={method:metrics(name) for method,name in names.items()}
        a,b=values['Static'],values['Token']
        checks=dict(output=all(v['output_gate']['passed'] for v in values.values()),
            P_rows=a['work_accounting']['P_GEMM_token_layer_rows']==b['work_accounting']['P_GEMM_token_layer_rows'],
            D_rows=a['work_accounting']['D_GEMM_token_layer_rows']==b['work_accounting']['D_GEMM_token_layer_rows'],
            KV_bytes=a['work_accounting']['valid_KV_payload_bytes']==b['work_accounting']['valid_KV_payload_bytes'],
            source_groups=a['source_groups']==b['source_groups'],
            destination_groups=a['destination_groups']==b['destination_groups'])
        assert all(checks.values()),checks
        pairs.append(dict(seed=seed,names=names,checks=checks,values=values))
    comparisons=[]
    if pairs:
        for reference,candidate in (('Token','Static'),('Native','Static'),('Native','Token')):
            result={key:ratio_interval([p['values'][reference][key] for p in pairs],
                [p['values'][candidate][key] for p in pairs]) for key in keys}
            result['SLO_output_tokens_s']=ratio_interval(
                [p['values'][reference]['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'] for p in pairs],
                [p['values'][candidate]['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'] for p in pairs])
            result['Joint_attainment']=ratio_interval(
                [p['values'][reference]['SLO_regimes']['Decode_priority']['Joint_attainment'] for p in pairs],
                [p['values'][candidate]['SLO_regimes']['Decode_priority']['Joint_attainment'] for p in pairs])
            comparisons.append(dict(reference=reference,candidate=candidate,metrics=result,
                Decode_5pct_noninferiority=len(pairs)>=3 and all(result[k]['ratio_CI95'] is not None and
                    result[k]['ratio_CI95'][1]<=1.05 for k in ('ITL_p99_ms','TPOT_p95_ms'))))
    write(ROOT/'analysis/matched_Last12_independent_start_statistics.json',dict(
        starts=len(pairs),pairs=pairs,comparisons=comparisons,evidence='MEASURED',
        scope='New independent arrivals over a declared repeated64-document discovery pool. This is not a new-prompt holdout or an independently best-tuned token split. Match actual GEMM token-layer rows, logical KV bytes and source/destination group membership. Attention work differs structurally. Extra Layer hidden/residual bytes are not matched. Throughput is finite-cohort drain throughput, not sustainable capacity.'))
    print('MATCHED_PAIRED_STARTS',len(pairs),flush=True)


if __name__=='__main__':
    main()
