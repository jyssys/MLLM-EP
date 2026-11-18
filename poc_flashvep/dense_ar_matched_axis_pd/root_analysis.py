"""Separate structural work, host path identities and operator measurements."""
import collections
import json
import statistics
from .common import ROOT,write
from .host_paths import analyze,summary
from ..dense_ar_layer_token_pd.statistics_report import ratio_interval


def micro():
    path=ROOT/'raw/isolated_BF16_axis_geometry_v1/operator_geometry.json'
    if not path.exists():
        return dict(evidence='NOT_MEASURED')
    data=json.loads(path.read_text())
    supplement=ROOT/'raw/isolated_realized_axis_geometry_v1/operator_geometry.json'
    if supplement.exists():
        added=json.loads(supplement.read_text())
        # Keep each source start. Do not pool these samples into a new CI.
        data['supplement_results']=added['results']
    plan=json.loads((ROOT/'analysis/matched_axis_structural_work.json').read_text())['rows']
    values=[]
    for device in (0,1):
        samples=[r for r in data['results'] if r['GPU']==device]
        def projection_sum(rows):
            selected=[r for r in samples if r['kind']=='GEMM' and r['M']==rows]
            assert len(selected)==4
            return sum(r['p50_ms'] for r in selected)
        def attention(query,batch,keys=8000):
            selected=[r for r in samples if r['kind']=='ATTENTION' and r['query_tokens']==query and r['batch']==batch and r['KV_tokens']==keys]
            assert len(selected)==1
            return selected[0]['p50_ms']
        for pair in plan:
            m=pair['layers'];hist={int(s):n for s,n in pair['token_suffix_histogram'].items()}
            n=sum(hist.values())
            singleton_token=36*sum(count*projection_sum(suffix) for suffix,count in hist.items())/n
            layer=m*projection_sum(8000)
            token_attention=36*sum(count*attention(suffix,1) for suffix,count in hist.items())/n
            layer_attention=m*attention(8000,1)
            batch={4:8,8:4,12:3}[m]
            # These are independent operator medians, not serial request paths.
            grouped=36*sum(count*projection_sum(suffix*batch)/batch for suffix,count in hist.items())/n
            grouped_attention=36*sum(count*attention(suffix,batch)/batch for suffix,count in hist.items())/n
            row=dict(GPU=device,layers=m,modeled_Layer_GEMM_ms=layer,
                modeled_Token_singleton_GEMM_ms=singleton_token,
                modeled_Token_grouped_GEMM_ms=grouped,Token_group_size=batch,
                modeled_Layer_Attention_ms=layer_attention,
                modeled_Token_singleton_Attention_ms=token_attention,
                modeled_Token_grouped_Attention_ms=grouped_attention,
                Layer_over_Token_singleton_GEMM=layer/singleton_token,
                Layer_over_Token_grouped_GEMM=layer/grouped,
                Layer_over_Token_singleton_Attention=layer_attention/token_attention,
                Layer_over_Token_grouped_Attention=layer_attention/grouped_attention)
            if data.get('supplement_results'):
                extra=[r for r in data['supplement_results'] if r['GPU']==device and r['kind']=='ATTENTION']
                def source_attention(query):
                    selected=[r for r in extra if r['batch']==1 and r['query_tokens']==query and r['KV_tokens']==query]
                    assert len(selected)==1
                    return selected[0]['p50_ms']
                row.update(modeled_Layer_P_Attention_ms=(36-m)*source_attention(8000),
                    modeled_Token_P_Attention_ms=36*sum(count*source_attention(8000-suffix) for suffix,count in hist.items())/n)
            values.append(row)
    return dict(rows=values,profiles=[dict(label=r['label'],evidence=r['evidence'],
        observer_gate_5pct=r.get('observer_gate_5pct'),ratio=r.get('observed_over_control')) for r in data['profiles']],
        evidence='STRUCTURAL_ACCOUNTING',measurement_class='ISOLATED_MEDIANS_WEIGHTED_BY_LAYER_COUNTS',
        projection_API='torch.nn.functional.linear',faithful_Serving_projection_API=False,
        superseded_for_Serving_projection_model_by='analysis/actual_projection_API_geometry_model.json',
        scope='Sum four representative layer0 projection medians and multiply by actual layer counts. Attention is exact contiguous FA3 with synthetic operands. These totals exclude operator fusion, other layers, paged KV, handoff, scheduling and contention. They are modeled work components, not measured online suffix latency or serving speedup.')


def main():
    statistics_path=ROOT/'analysis/independent_start_statistics.json'
    data=json.loads(statistics_path.read_text()) if statistics_path.exists() else dict(starts=[],comparisons=[])
    starts=data['starts']
    host=[]
    for row in starts:
        if row['method']=='Agg':
            continue
        path=ROOT/'analysis'/(row['name']+'_host_paths.json')
        record=json.loads(path.read_text()) if path.exists() else analyze(row['name'])
        serialized=all(current['D_host_submit']>=previous['last_worker_notice']
            for previous,current in zip(record['cohorts'],record['cohorts'][1:]))
        positive=all(r['preparation_host_ms']>=0 and r['worker_decode_residency_ms']>=0 for r in record['cohorts'])
        guard_ms=(record['summed_cohort_preparation_per_completed_request_ms']+
            record['summed_cohort_decode_residency_per_completed_request_ms'])
        host.append(dict(stage=row['stage'],method=row['method'],layers=row['layers'],
            count=row['count'],cap=row['cap'],QPS=row['QPS'],seed=row['seed'],
            admission=row['admission'],name=row['name'],cohort_sizes=record['cohort_sizes'],
            host_path_mean_ms={key:value['mean'] for key,value in record['summaries'].items()},
            mean_TTFT_ms=row['mean_TTFT_ms'],TTFT_p95_ms=row['TTFT_p95_ms'],
            summed_decode_residency_per_request_ms=record['summed_cohort_decode_residency_per_completed_request_ms'],
            summed_D_preparation_per_request_ms=record['summed_cohort_preparation_per_completed_request_ms'],
            serialized_D_host_spans_nonoverlap=serialized and positive,
            D_host_guard_span_per_request_ms=guard_ms,
            D_host_span_request_rate_ceiling=1000/guard_ms if serialized and positive and guard_ms>0 else None,
            actual_finite_cohort_completed_QPS=row['completed_QPS'],
            host_span_rate_scope='Count divided by the sum of nonoverlapping D-host-submit-to-last-worker-notice intervals. This is a finite-run accounting ceiling. It includes host/transfer waiting. It is not physical GPU capacity, exclusive compute, kernel busy, or sustainable serving capacity.',
            D_host_preparation_ms=record['preparation_ms'],
            transport=json.loads((ROOT/'analysis'/(row['name']+'_transport_audit.json')).read_text())))
    pairs=[]
    lookup={(r['stage'],r['method'],r['layers'],r['cap'],r['QPS'],r['seed'],r['count'],r['admission']):r for r in host}
    for key,layer in lookup.items():
        stage,method,m,cap,qps,seed,count,limit=key
        if method!='Layer':continue
        token=lookup.get((stage,'Token',m,cap,qps,seed,count,limit))
        if token is None:continue
        difference={k:layer['host_path_mean_ms'][k]-token['host_path_mean_ms'][k] for k in layer['host_path_mean_ms']}
        total=layer['mean_TTFT_ms']-token['mean_TTFT_ms']
        assert abs(sum(difference.values())-total)<1e-5
        pairs.append(dict(stage=stage,layers=m,cap=cap,QPS=qps,seed=seed,count=count,admission=limit,
            Layer=layer['name'],Token=token['name'],mean_TTFT_Layer_minus_Token_ms=total,
            host_mean_path_difference_ms=difference,
            largest_absolute_host_path_difference=max(difference,key=lambda k:abs(difference[k])),
            Layer_cohort_sizes=layer['cohort_sizes'],Token_cohort_sizes=token['cohort_sizes'],
            Layer_decode_residency_per_request_ms=layer['summed_decode_residency_per_request_ms'],
            Token_decode_residency_per_request_ms=token['summed_decode_residency_per_request_ms'],
            D_host_guard_span_Layer_over_Token=layer['D_host_guard_span_per_request_ms']/token['D_host_guard_span_per_request_ms'],
            actual_completed_rate_Token_over_Layer=token['actual_finite_cohort_completed_QPS']/layer['actual_finite_cohort_completed_QPS'],
            scope='An exact host-clock identity. A path difference is not proof of an exclusive compute cause. Stage p95 values are not added.'))
    crossover=[]
    grouped=collections.defaultdict(list)
    for row in data.get('comparisons',[]):
        if row['reference']=='Token' and row['candidate']=='Layer':
            grouped[(row['stage'],row['layers'],row['QPS'],row['count'],row['admission'])].append(row)
    for (stage,m,qps,count,limit),rows in grouped.items():
        layer_wins=[];token_wins=[];uncertain=[]
        for row in sorted(rows,key=lambda r:r['cap']):
            ci=row['metrics']['TTFT_p95_ms']['ratio_CI95']
            if row['starts']>=3 and ci and ci[1]<1:
                layer_wins.append(row['cap'])
            elif row['starts']>=3 and ci and ci[0]>1:
                token_wins.append(row['cap'])
            else:
                uncertain.append(row['cap'])
        verdict=('GENUINE_CROSSOVER_TTFT_CONFIRMED' if layer_wins and token_wins
                 else 'LAYER_FAVORED_ON_CONFIRMED_POINTS' if layer_wins
                 else 'TOKEN_FAVORED_ON_CONFIRMED_POINTS' if token_wins else 'INCONCLUSIVE')
        crossover.append(dict(stage=stage,layers=m,QPS=qps,count=count,admission=limit,
            verdict=verdict,Layer_favored_outputs=layer_wins,Token_favored_outputs=token_wins,
            uncertain_outputs=uncertain,
            discovery_TTFT_ratio_by_output={str(r['cap']):r['metrics']['TTFT_p95_ms']['ratio'] for r in rows},
            scope='A TTFT crossover is distinct from Native-relative decode protection and absolute SLO success. Confidence intervals are pointwise. No global all-workload dominance claim.'))
    write(ROOT/'analysis/root_cause_components.json',dict(host_starts=host,paired_host_differences=pairs,
        crossover=crossover,isolated=micro(),evidence='TRACE_ANALYSIS',
        actual_Serving_projection_model='analysis/actual_projection_API_geometry_model.json',
        crossover_CI_convention='Original Type7 linear quantiles of paired-start bootstrap draws. Read the separate inverse-CDF sensitivity audit before a strong research verdict.',
        scope='Keep structural work, isolated operator models, same-run kernels and client-visible performance separate. Matched cap8 is not matched actual cohort geometry. The common cap1 diagnostic tests this runtime interaction. It does not establish a universal axis ranking.'))
    print('ROOT_COMPONENTS',len(host),len(pairs),len(crossover),flush=True)


if __name__=='__main__':
    main()
