"""Keep mean timing identities separate from causal interventions."""
import collections
import json
import statistics
import time
from .common import ROOT,PREVIOUS,write
from .resampling import mean_interval,ratio_interval
from .model_validation import extract

SPAN_KEYS=('arrival_to_P_host_submit','P_submit_to_source_notification',
    'source_notification_to_D_host_submit','D_submit_to_first_worker_notice','first_worker_notice_to_client')


def window_intervention(method,layers,seed=2026111901):
    names={w:f'source_window_control_{method}_Last{layers}_N72_O16_q6_s{seed}_B2_W{w}_v1' for w in (32,4)}
    paths={w:ROOT/'analysis'/(n+'_service_queue_audit.json') for w,n in names.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    values={w:json.loads(p.read_text()) for w,p in paths.items()}
    return dict(method=method,layers=layers,seed=seed,names=names,starts_per_setting=1,
        absolute={str(w):{k:v[k] for k in ('P_submit_to_ready_mean_ms','D_host_slot_per_request_ms',
            'Ready_D_mean_ms','P_external_wait_mean_ms')} for w,v in values.items()},
        W4_minus_W32_ms={k:values[4][k]-values[32][k] for k in
            ('P_submit_to_ready_mean_ms','D_host_slot_per_request_ms','Ready_D_mean_ms','P_external_wait_mean_ms')},
        TTFT_mean_ratio_W4_over_W32=values[4]['metric_summary']['mean_TTFT_ms']/values[32]['metric_summary']['mean_TTFT_ms'],
        TTFT_p95_ratio_W4_over_W32=values[4]['metric_summary']['TTFT_p95_ms']/values[32]['metric_summary']['TTFT_p95_ms'],
        output_token_rate_ratio_W4_over_W32=values[4]['metric_summary']['output_tokens_s']/values[32]['metric_summary']['output_tokens_s'],
        reservation_rejections={str(w):v['reservation_rejections'] for w,v in values.items()},
        common_credit_retries={str(w):v['common_credit_retries'] for w,v in values.items()},
        evidence='MEASURED_ONLINE',
        scope='Paired source-window intervention, not a compute or admission-batch change. One start per setting cannot establish equivalence. Inspect service drift before treating this as a pure relocation of waiting.')


def window_summary(rows):
    groups=collections.defaultdict(list)
    for row in rows:
        groups[row['method'],row['layers']].append(row)
    result=[]
    for (method,m),values in sorted(groups.items()):
        absolute={w:[r['absolute'][str(w)] for r in values] for w in (32,4)}
        result.append(dict(method=method,layers=m,starts=len(values),
            seeds=[r['seed'] for r in values],
            mean_delta_ms={k:dict(mean=statistics.mean(r['W4_minus_W32_ms'][k] for r in values),
                CI95=mean_interval([r['W4_minus_W32_ms'][k] for r in values]))
                for k in values[0]['W4_minus_W32_ms']},
            host_service_ratios={k:ratio_interval([r[k] for r in absolute[32]],
                [r[k] for r in absolute[4]]) for k in
                ('P_submit_to_ready_mean_ms','D_host_slot_per_request_ms')},
            outcome_ratios={k:dict(mean=statistics.mean(r[k] for r in values),
                CI95=mean_interval([r[k] for r in values])) for k in
                ('TTFT_mean_ratio_W4_over_W32','TTFT_p95_ratio_W4_over_W32',
                 'output_token_rate_ratio_W4_over_W32')},
            CI_unit='Paired server starts. Outcome mean is the mean paired ratio. Host-service ratio is the ratio of paired means.',
            evidence='MEASURED_ONLINE',
            scope='The original seed1901 informed this intervention. Seeds1950/1951 are independent arrival confirmations. Three starts provide limited tail precision.'))
    return result


def new_mean_identities():
    path=ROOT/'analysis/independent_start_statistics.json'
    if not path.exists():
        return []
    starts=json.loads(path.read_text())['starts']
    groups=collections.defaultdict(dict)
    for r in starts:
        if r['method'] not in ('Layer','Token'):
            continue
        key=(r['stage'],r['layers'],r['output'],r['QPS'],r['count'],
             r['actual_batch'],r['source_window'],r['seed'])
        groups[key][r['method']]=r
    output=[]
    for key,pair in sorted(groups.items()):
        if set(pair)!=set(('Layer','Token')):
            continue
        hosts={a:json.loads((ROOT/'analysis'/(r['name']+'_host_paths.json')).read_text())
            for a,r in pair.items()}
        delta={span:hosts['Layer']['summaries'][span]['mean']-
            hosts['Token']['summaries'][span]['mean'] for span in SPAN_KEYS}
        total=pair['Layer']['metric_summary']['mean_TTFT_ms']-pair['Token']['metric_summary']['mean_TTFT_ms']
        assert abs(sum(delta.values())-total)<1e-5
        output.append(dict(stage=key[0],layers=key[1],output=key[2],QPS=key[3],count=key[4],
            batch=key[5],window=key[6],seed=key[7],names={a:r['name'] for a,r in pair.items()},
            Layer_minus_Token_mean_TTFT_ms=total,mean_span_differences_ms=delta,
            evidence='TRACE_ANALYSIS',
            scope='An exact client-mean identity. Correlated stage differences are not independent causal effects.'))
    return output


def identity_summary(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['stage'],r['layers'],r['output'],r['QPS'],r['batch'],r['window']].append(r)
    return [dict(stage=k[0],layers=k[1],output=k[2],QPS=k[3],batch=k[4],window=k[5],
        starts=len(v),seeds=[r['seed'] for r in v],
        mean_TTFT_difference_ms=statistics.mean(r['Layer_minus_Token_mean_TTFT_ms'] for r in v),
        mean_spans_ms={field:dict(mean=statistics.mean(r['mean_span_differences_ms'][field] for r in v),
            CI95=mean_interval([r['mean_span_differences_ms'][field] for r in v])) for field in SPAN_KEYS},
        evidence='TRACE_ANALYSIS',scope='Paired new independent starts. Mean-span identities do not decompose p95 or establish independent causal effects.')
        for k,v in sorted(groups.items())]


def output_intervention():
    groups=collections.defaultdict(dict)
    for r in json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())['starts']:
        if r['stage']=='output_residency_counterfactual' and r['method'] in ('Layer','Token'):
            groups[r['method'],r['layers'],r['QPS'],r['actual_batch'],r['source_window'],r['seed']][r['output']]=r
    paired=[]
    for key,rows in sorted(groups.items()):
        if set(rows)!={16,64}:continue
        assert rows[16]['arrival_offsets']==rows[64]['arrival_offsets']
        assert rows[16]['dataset']==rows[64]['dataset'] and rows[16]['count']==rows[64]['count']
        values={}
        for output,r in rows.items():
            audit=json.loads((ROOT/'analysis'/(r['name']+'_service_queue_audit.json')).read_text())
            slot=json.loads((ROOT/'analysis'/(r['name']+'_slot_decomposition.json')).read_text())
            feedback=json.loads((ROOT/'analysis'/(r['name']+'_reservation_feedback.json')).read_text())
            values[output]=dict(P_service_ms=audit['P_submit_to_ready_mean_ms'],
                P_backlogged_gap_ms=feedback['mean_backlogged_nonwindow_P_gap_ms'],
                P_backlogged_ACK_ms=feedback['mean_backlogged_nonwindow_gap_parts_ms']['reservation_RTT_ms'],
                D_preparation_per_request_ms=slot['mean_D_per_request_ms']['D_submit_to_all_first_worker_ms'],
                D_decode_residence_per_request_ms=slot['mean_D_per_request_ms']['all_first_to_last_worker_ms'],
                D_delivery_per_request_ms=slot['mean_D_per_request_ms']['last_worker_to_last_client_ms'],
                D_guard_per_request_ms=audit['D_host_slot_per_request_ms'],
                Ready_D_wait_ms=audit['Ready_D_mean_ms'],P_external_wait_ms=audit['P_external_wait_mean_ms'])
        assert all(v is not None for group in values.values() for v in group.values())
        paired.append(dict(method=key[0],layers=key[1],QPS=key[2],batch=key[3],window=key[4],seed=key[5],
            names={str(o):r['name'] for o,r in rows.items()},absolute_ms=values,
            Output64_minus16_ms={field:values[64][field]-values[16][field] for field in values[16]}))
    summaries=[]
    for a in ('Layer','Token'):
        selected=[r for r in paired if r['method']==a]
        if not selected:continue
        summaries.append(dict(method=a,starts=len(selected),seeds=[r['seed'] for r in selected],
            mean_delta_ms={field:dict(mean=statistics.mean(r['Output64_minus16_ms'][field] for r in selected),
                CI95=mean_interval([r['Output64_minus16_ms'][field] for r in selected]))
                for field in selected[0]['Output64_minus16_ms']}))
    return dict(paired=paired,summary=summaries,evidence='MEASURED_ONLINE_AND_TRACE_ANALYSIS',
        scope='Same input, partition, offered QPS, arrival seed, request count and actual1/1/2. Output cap changes16→64. Runtime context cap derives from that cap,8016→8064. This is the total intervention, not an isolated GPU Decode-duration effect. Host phase identities are not exclusive GPU times.')


def historical_amplification(starts):
    """Explain completed traces; do not call fitted-after-run costs predictions."""
    result=[]
    for m in (4,8,12):
        selected=[r for r in starts if r['layers']==m]
        for seed in sorted({r['seed'] for r in selected}):
            pair={r['method']:r for r in selected if r['seed']==seed}
            if set(pair)!=set(('Layer','Token')):continue
            effective={}
            for a,r in pair.items():
                cfg,_,c=extract(PREVIOUS,r['name']);b=cfg['D_admission_batch']
                p=c['P']+c['P_gap'];d=c['preparation']+(c['decode']+c['D_gap'])/b
                effective[a]=dict(P_effective_ms=1000*p,D_effective_ms_per_request=1000*d,
                    cycle_model_ms_per_request=1000*max(p,d))
            delta=effective['Layer']['cycle_model_ms_per_request']-effective['Token']['cycle_model_ms_per_request']
            measured=pair['Layer']['TTFT_p95_ms']-pair['Token']['TTFT_p95_ms']
            rank=.95*(cfg['count']-1)
            result.append(dict(layers=m,seed=seed,count=cfg['count'],batch=b,
                effective_demands=effective,Layer_minus_Token_cycle_ms_per_request=delta,
                tail_rank_approximation=rank,cycle_delta_times_tail_rank_ms=rank*delta,
                measured_Layer_minus_Token_TTFT_p95_ms=measured,evidence='TRACE_ANALYSIS',
                scope='Post-run explanatory proxy, using measured service costs from the same completed trace. Max-stage cycle times tail rank is a deterministic overloaded-pipeline approximation, not an M/M/1 formula, independent causal effect, held-out prediction or measured GPU time. The event-driven replay has separate discovery-to-confirmation validation.'))
    return result


def main():
    old=json.loads((PREVIOUS/'analysis/independent_start_statistics.json').read_text())['starts']
    old=[r for r in old if r['stage']=='cohort2_confirmation' and r['method'] in ('Layer','Token')]
    output=[]
    for m in (4,8,12):
        selected=[r for r in old if r['layers']==m]
        seeds=sorted({r['seed'] for r in selected})
        paired=[]
        for seed in seeds:
            pair={r['method']:r for r in selected if r['seed']==seed}
            assert set(pair)==set(('Layer','Token'))
            hosts={a:json.loads((PREVIOUS/'analysis'/(r['name']+'_host_paths.json')).read_text()) for a,r in pair.items()}
            delta={key:hosts['Layer']['summaries'][key]['mean']-hosts['Token']['summaries'][key]['mean'] for key in SPAN_KEYS}
            mean_delta=pair['Layer']['mean_TTFT_ms']-pair['Token']['mean_TTFT_ms']
            assert abs(sum(delta.values())-mean_delta)<1e-5
            paired.append(dict(seed=seed,names={a:r['name'] for a,r in pair.items()},
                Layer_minus_Token_mean_TTFT_ms=mean_delta,mean_span_differences_ms=delta,
                Layer_minus_Token_TTFT_p95_ms=pair['Layer']['TTFT_p95_ms']-pair['Token']['TTFT_p95_ms']))
        output.append(dict(layers=m,starts=len(paired),paired=paired,
            mean_TTFT_difference_ms=statistics.mean(r['Layer_minus_Token_mean_TTFT_ms'] for r in paired),
            mean_spans_ms={k:dict(mean=statistics.mean(r['mean_span_differences_ms'][k] for r in paired),
                CI95=mean_interval([r['mean_span_differences_ms'][k] for r in paired])) for k in SPAN_KEYS},
            evidence='TRACE_ANALYSIS',scope='Exact identity for mean client TTFT only. Do not add stage-p95 differences or infer causal independence of queue and service terms.'))
    new=[r for m in (4,8,12) for a in ('Layer','Token') for seed in (2026111901,2026111950,2026111951)
        if (r:=window_intervention(a,m,seed)) is not None]
    identities=new_mean_identities()
    write(ROOT/'analysis/service_queue_causal_decomposition.json',dict(time=time.time(),
        historical_mean_identities=output,new_mean_identities=identities,new_mean_identity_summary=identity_summary(identities),
        output_cap_intervention=output_intervention(),
        historical_queue_amplification_proxy=historical_amplification(old),
        source_window_interventions=new,source_window_independent_start_summary=window_summary(new),
        exclusive_GPU_components='NOT_MEASURED',physical_GPU_idle='NOT_MEASURED',
        scope='Timing identities explain where a difference occurs. Interventions are needed to test why. Keep historical and new observations separate.'))
    for row in new:
        print('SOURCE_WINDOW_RESULT',row['method'],row['layers'],row['W4_minus_W32_ms'],
            row['TTFT_p95_ratio_W4_over_W32'],row['output_token_rate_ratio_W4_over_W32'],flush=True)


if __name__=='__main__':
    main()
