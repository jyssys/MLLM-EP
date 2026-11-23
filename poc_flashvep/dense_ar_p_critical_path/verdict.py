"""Derive scoped scientific verdicts without treating completion as success."""
import json
import time
from .common import ROOT,bind,write,digest

FAMILIES=(('Credit Retirement','c3_credit_confirmation'),('Partition Geometry','p_geometry_confirmation'),
    ('D Cap4','p_geometry_D_cap4_confirmation'),('Common Coordinator','coordinator_hygiene_confirmation'),
    ('Stable-source Stock Coordinator','coordinator_hygiene_stable_stock_confirmation'),
    ('Fixed-port Coordinator','coordinator_hygiene_fixed_ports_confirmation'),
    ('Warm-regime Coordinator','coordinator_hygiene_warm_regime_confirmation'),
    ('Output192 D cap2→4','output192_D_cap_intervention'),('Fixed-output Stress','p_output_stress'))


def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else None


def analyze():
    bind()
    from ..dense_ar_output_aware_v2.statistics import paired_difference
    outcomes=[];host=[];sources={}
    for family,stem in FAMILIES:
        path=ROOT/'analysis'/(stem+'_analysis.json')
        if not path.exists():continue
        sources[family]=digest(path);data=json.loads(path.read_text())
        cells={(row['mode'],row['label']):row for row in data['cells']}
        for row in data['comparisons']:
            if row['right']!='StockToken':continue
            cell=cells[row['mode'],row['left']];ttft=row['ratios']['TTFT_p95_ms'];gp=row['ratios']['SLO_qualified_output_tokens_per_second']
            enough=row['paired_starts']>=3 and ttft['CI95'] is not None and gp['CI95'] is not None
            extra=cell.get('additional_decode_gates',{})
            safety=cell['decode_gate'].get('qualified',False) and all(value.get('qualified',False) for value in extra.values())
            q=row['quality_passed_all_paired_starts']
            tw=enough and ttft['CI95'][1]<1;gw=enough and gp['CI95'][0]>1
            runtime_pass=row.get('runtime_policy_passed_all_paired_starts',True)
            serving=enough and q and runtime_pass and safety and tw and gw
            outcomes.append(dict(family=family,mode=row['mode'],label=row['left'],paired_starts=row['paired_starts'],
                TTFT_ratio=ttft,goodput_ratio=gp,quality_passed=q,decode_passed=safety,
                runtime_policy_passed=runtime_pass,
                statistically_confirmed_TTFT_win=tw,statistically_confirmed_goodput_win=gw,serving_GO=serving,
                identical_per_request_output_work=all(row['per_start_request_output_work_exact']),
                decision='GO' if serving else 'HOLD_INSUFFICIENT_STARTS' if not enough else 'NO_GO_ON_THIS_CELL',
                scope='Pointwise paired intervals. A Natural EOS win can have different generated work. '
                    'This decision does not establish native-relative improvement or sustainable capacity.'))
        starts={(row['mode'],row['start'],row['label']):row for row in data['starts']}
        for mode in dict.fromkeys(row['mode'] for row in data['starts']):
            pairs=[(row,starts.get((mode,row['start'],'StockToken'))) for row in data['starts']
                if row['mode']==mode and row['label']=='OriginalLayer']
            pairs=[pair for pair in pairs if pair[1] is not None]
            if not pairs:continue
            p=lambda row:sum(row['mean_host_path_ms'][key] for key in ('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms','P_prepare_to_ready_ms'))
            d=lambda row:sum(row['mean_host_path_ms'][key] for key in ('Ready_to_D_submit_ms','D_submit_to_accept_ms','D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms'))
            result=dict(family=family,mode=mode,paired_starts=len(pairs),
                P_path_delta_ms=paired_difference([p(a) for a,b in pairs],[p(b) for a,b in pairs]),
                post_P_path_delta_ms=paired_difference([d(a) for a,b in pairs],[d(b) for a,b in pairs]),
                mean_TTFT_delta_ms=paired_difference([a['mean_host_path_ms']['TTFT_ms'] for a,b in pairs],
                    [b['mean_host_path_ms']['TTFT_ms'] for a,b in pairs]))
            assert abs(result['P_path_delta_ms']['estimate']+result['post_P_path_delta_ms']['estimate']-result['mean_TTFT_delta_ms']['estimate'])<1e-5
            host.append(result)
    norm=[]
    for axis in ('layer','hybrid'):
        value=read('state/p_norm_ir_'+axis+'_strict8.json')
        if value:
            norm.append(dict(axis=axis,passed=value['passed'],requests=len(value.get('rows',[])),
                exact_KV=sum(row['KV_exact'] for row in value.get('rows',[])),
                exact_hidden=sum(row['hidden_exact'] for row in value.get('rows',[])),
                exact_greedy=sum(row['greedy_exact'] for row in value.get('rows',[])),
                primary_performance='NOT_RUN' if not value['passed'] else 'REQUIRES_FRESH_32_400_AND_N3',
                scope='Same-family provider replacement gate. Equal greedy tokens do not erase KV/hidden differences. '
                    'A strict failure is not itself proof of aggregate task-quality loss.'))
    summary=dict(time=time.time(),sources_SHA256=sources,outcomes=outcomes,additive_mean_host_comparisons=host,norm_probe=norm,
        dynamic_insertion='NOT_RUN',safe_window='NOT_VERIFIED',root_cause='PARTIAL_ROOT_CAUSE_IDENTIFICATION',
        scope='Scientific outcome is separate from operational completion. Keep failed gates,negative cells and NOT_RUN. '
            'The historical1.76-second host difference cannot be retroactively assigned exclusive GPU/host causal percentages. '
            'Matched-work attention structure and new kernel activity support a compute/queue contribution. '
            'Credit,geometry,admission and coordinator interventions remain separate families. '
            'No successful isolated or offline result is promoted to continuous-serving success.')
    write(ROOT/'analysis/scientific_verdict.json',summary)
    from .critical_analysis import csv_save
    csv_save(ROOT/'figures/scientific_verdict_cells.csv',outcomes)
    csv_save(ROOT/'figures/additive_mean_host_comparisons.csv',host)
    print('SCIENTIFIC_VERDICT',len(outcomes),'cells',sum(row['serving_GO'] for row in outcomes),'GO',flush=True)
    return summary


if __name__=='__main__':analyze()
