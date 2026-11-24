"""Compare paired host paths and bounded frozen-trace metric headroom."""
import collections
import csv
import json
import time
import numpy as np
from .common import ROOT,REPO,initialize,preserve,write,digest
from .reproduction import records
from .request_trace import STAGES
from .failure_analysis import save_csv
from ..dense_ar_output_aware_v2.statistics import ratio,paired_difference


def read_csv(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def minimum_fraction(ttft,envelope,target):
    """Find a frozen-envelope p95 target, without changing later queues."""
    a=np.asarray(ttft,float);cost=np.asarray(envelope,float)
    assert a.shape==cost.shape and len(a)>0 and np.all(cost>=-1e-3)
    cost=np.maximum(cost,0.)
    evaluate=lambda fraction:float(np.percentile(np.maximum(a-fraction*cost,0.),95))
    if evaluate(0.)<=target:return dict(reachable=True,fraction=0.,target_p95_ms=target)
    if evaluate(1.)>target:return dict(reachable=False,fraction=None,target_p95_ms=target,
        p95_after_full_removal_ms=evaluate(1.))
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if evaluate(mid)<=target:hi=mid
        else:lo=mid
    return dict(reachable=True,fraction=hi,target_p95_ms=target,
        p95_after_removal_ms=evaluate(hi),mean_removed_envelope_ms=float(np.mean(cost)*hi))


def descriptive_correlation(x,y):
    """Return descriptive coefficients only, with average ranks for ties."""
    a,b=np.asarray(x,float),np.asarray(y,float)
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return dict(n=len(a),Pearson=None,Spearman=None)
    def ranks(values):
        _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
        average=np.cumsum(counts)-counts+(counts-1)/2
        return average[inverse]
    return dict(n=len(a),Pearson=float(np.corrcoef(a,b)[0,1]),
        Spearman=float(np.corrcoef(ranks(a),ranks(b))[0,1]))


def analyze():
    initialize();preserve()
    failure=json.loads((ROOT/'analysis/request_failure_attribution.json').read_text())
    paths=json.loads((ROOT/'analysis/request_critical_paths.json').read_text())
    metric={(r['family'],r['label'],r['start']):r for r in failure['starts']}
    host={(r['family'],r['label'],r['start']):r for r in paths['starts']}
    grouped=collections.defaultdict(dict)
    for key,row in metric.items():grouped[key[0]][key[1],key[2]]=row
    comparisons=[]
    for family,items in sorted(grouped.items()):
        labels=sorted({label for label,index in items})
        pairs=[(label,reference) for reference in ('NativeStrong','NativeCommon0','NativeCommon2','StockToken') for label in labels
            if label!=reference and label not in ('Agg','Deflection') and reference in labels]
        pairs.extend((a,b) for a,b in (('OptimizedLayer','OriginalLayer'),('OptimizedHybrid','OriginalHybrid'),
            ('OriginalHybrid','HybridGCControl'),('StockCap2','StockToken'),('StockCap2RegOpen','StockCap2'),
            ('NativeRegOpen','NativeCommon2'),('LayerRegOpen','Layer'),('HybridRegOpen','Hybrid'),
            ('LayerRetirement','Layer'),('HybridRetirement','Hybrid')) if a in labels and b in labels)
        pairs=list(dict.fromkeys(pairs))
        for left,right in pairs:
            indices=sorted({index for label,index in items if label==left and (right,index) in items})
            actual=[(items[left,index],items[right,index]) for index in indices]
            a,b=zip(*actual)
            for x,y in actual:
                for key in ('seed','GPU_pair','dataset_SHA256','mode'):assert x[key]==y[key],(family,left,right,key)
            stages={}
            if all((family,left,i) in host and (family,right,i) in host for i in indices):
                for field in (*STAGES,'Arrival_to_P_ready_ms','post_P_to_first_client_ms'):
                    stages[field]=paired_difference([host[family,left,i]['means'][field] for i in indices],
                        [host[family,right,i]['means'][field] for i in indices])
            comparisons.append(dict(family=family,left=left,right=right,paired_starts=len(indices),
                ratios={field:ratio([x[field] for x in a],[x[field] for x in b]) for field in (
                    'TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second','SLO_qualified_output_tokens_per_second')},
                differences={field:paired_difference([x[field] for x in a],[x[field] for x in b]) for field in (
                    'TTFT_mean_ms','TTFT_SLO_attainment','every_ITL_SLO_attainment','Joint_SLO_attainment')},
                mean_host_path_delta_ms=stages,
                scope='Paired independent-start empirical intervals are pointwise and unadjusted. Host mean deltas do not decompose p95 or GPU operators. '
                    'Natural output work can differ. Different source/cache families are not pooled.'))
    # Join request failures to exact host boundaries. Do not regress one arm against another unmatched request.
    failures=read_csv(ROOT/'figures/request_slo_failures.csv')
    traces=read_csv(ROOT/'figures/request_critical_paths.csv')
    lookup={(r['family'],r['name'],int(r['sample_id'])):r for r in failures}
    stage_groups={
        'Coordinator_wait':('Arrival_to_P_submit_ms',),
        'P_host_admission_envelope':('P_submit_to_accept_ms','P_accept_to_prepare_ms'),
        'P_prepare_to_ready_envelope':('P_prepare_to_ready_ms',),
        'P_ready_to_D_submit':('Ready_to_D_submit_ms',),
        'D_registration_envelope':('D_submit_to_accept_ms',),
        'D_accept_to_prepare_envelope':('D_accept_to_prepare_ms',),
        'D_prepare_to_first_worker_envelope':('D_prepare_to_first_worker_ms',),
        'Client_delivery':('Worker_to_client_ms',)}
    joined=[];oracles=[];dist=[];paired_requests=[];native_targets=[]
    by_run=collections.defaultdict(list)
    for row in traces:
        key=(row['family'],row['name'],int(row['sample_id']));client=lookup[key]
        assert row['prompt_id']==client['prompt_id']
        combined=dict(row,TTFT_PASS=client['TTFT_PASS']=='True',ITL_PASS=client['ITL_PASS']=='True',
            TPOT_PASS=client['TPOT_PASS']=='True',JOINT_PASS=client['JOINT_PASS']=='True',
            category=client['category'],input_bucket=client['input_bucket'],output_bucket=client['output_bucket'])
        combined.update({name:sum(float(row[field]) for field in fields) for name,fields in stage_groups.items()})
        remote=json.loads(row['remote_read_breakdown'])
        if remote.get('ordered'):
            for field in ('D_accept_to_read_start_ms','read_host_enclosure_ms','read_end_to_receive_ACK_ms','receive_ACK_to_D_prepare_ms'):
                combined[field]=remote[field]
        joined.append(combined);by_run[row['family'],row['name']].append(combined)
    for (family,name),rows in by_run.items():
        label=rows[0]['label'];index=int(rows[0]['start']);meta=dict(family=family,name=name,label=label,start=index)
        observed=metric[family,label,index];seconds=observed['measurement_seconds']
        for threshold in (500,1000,2000,3000,4000,4500,5000,5500,6000,8000,10000,15000,20000):
            good=[row for row in rows if float(row['TTFT_ms'])<=threshold]
            dist.append(dict(meta,threshold_ms=threshold,requests_at_or_below=len(good),request_count=len(rows),
                CDF=len(good)/len(rows),output_tokens_at_or_below=sum(int(row['output_tokens']) for row in good)))
        # These are frozen-path removals. Later queue states and cohort completion are not simulated.
        groups=dict(stage_groups)
        groups.update(P_all_pre_ready=tuple(STAGES[:4]),D_all_post_ready=tuple(STAGES[4:]),
            All_host_path=tuple(STAGES))
        for field in ('D_accept_to_read_start_ms','read_host_enclosure_ms','read_end_to_receive_ACK_ms','receive_ACK_to_D_prepare_ms'):
            if all(field in row for row in rows):groups[field]=(field,)
        for group,fields in groups.items():
            native=metric.get((family,'NativeStrong',index))
            if native is not None and label!='NativeStrong':
                target=minimum_fraction([float(row['TTFT_ms']) for row in rows],
                    [sum(float(row[field]) for field in fields) for row in rows],native['TTFT_p95_ms'])
                native_targets.append(dict(meta,stage_group=group,**target,evidence='OFFLINE_FROZEN_TRACE_TARGET_REMOVAL',
                    scope='Remove a uniform fraction of one observed host envelope until this start matches NativeStrong p95. '
                        'Future queue states,ITLs,output work and cohort duration stay frozen. Reachable is a metric condition,not an implementability claim.'))
            for fraction in (.1,.25,.5,.75,1.):
                ttft=[max(0.,float(row['TTFT_ms'])-fraction*sum(float(row[field]) for field in fields)) for row in rows]
                pass_ids=[i for i,(row,value) in enumerate(zip(rows,ttft)) if value<=5000 and row['ITL_PASS'] and row['TPOT_PASS']]
                tokens=sum(int(rows[i]['output_tokens']) for i in pass_ids)
                oracles.append(dict(meta,stage_group=group,fraction_removed=fraction,
                    TTFT_p95_ms=float(np.percentile(ttft,95)),joint_attainment=len(pass_ids)/len(rows),
                    SLO_goodput_frozen_window=tokens/seconds,
                    recovered_requests=len(pass_ids)-sum(row['JOINT_PASS'] for row in rows),
                    recovered_output_tokens=tokens-round(observed['SLO_qualified_output_tokens_per_second']*seconds),
                    evidence='OFFLINE_FROZEN_TRACE_METRIC_ORACLE',
                    scope='Remove only this observed host-path envelope. Do not update later queues,KV,ITLs,output work or cohort duration. '
                        'The envelope can contain compute and waiting. This is not a queue simulator,achievable speedup or identified causal effect.'))
    matching=collections.defaultdict(dict)
    for row in joined:matching[row['family'],row['start'],row['sample_id']][row['label']]=row
    for (family,index,sample),labels in matching.items():
        for reference in ('NativeStrong','StockToken'):
            if reference not in labels:continue
            teacher=labels[reference]
            for label,row in labels.items():
                if label==reference:continue
                assert row['prompt_id']==teacher['prompt_id']
                paired_requests.append(dict(family=family,start=int(index),sample_id=int(sample),label=label,reference=reference,
                    prompt_id=row['prompt_id'],input_tokens=int(row['input_tokens']),
                    output_tokens=int(row['output_tokens']),reference_output_tokens=int(teacher['output_tokens']),
                    TTFT_delta_ms=float(row['TTFT_ms'])-float(teacher['TTFT_ms']),
                    P_path_delta_ms=float(row['Arrival_to_P_ready_ms'])-float(teacher['Arrival_to_P_ready_ms']),
                    post_P_delta_ms=float(row['post_P_to_first_client_ms'])-float(teacher['post_P_to_first_client_ms']),
                    candidate_JOINT_PASS=row['JOINT_PASS'],reference_JOINT_PASS=teacher['JOINT_PASS'],
                    same_actual_output_work=row['output_tokens']==teacher['output_tokens']))
    # An exact accounting identity for the observed goodput gap. It is not runtime attribution.
    identities=[]
    for main in ('Historical_Main400_Natural','Current_Main400_Natural','Current_Heldout192_EqualReplay'):
        for label in ('Layer','Hybrid','LayerRetirement','HybridRetirement','LayerRegOpen','HybridRegOpen'):
            indices=sorted(index for family,name,index in metric if family==main and name==label and (main,'StockToken',index) in metric)
            pairs=[]
            for index in indices:
                left,right=(metric[main,key,index] for key in (label,'StockToken'))
                a={r['category']:r for r in failure['categories'] if r['family']==main and r['label']==label and r['start']==index}
                b={r['category']:r for r in failure['categories'] if r['family']==main and r['label']=='StockToken' and r['start']==index}
                lost={key:a[key]['output_tokens_per_second']-b[key]['output_tokens_per_second'] for key in a if key!='JOINT_PASS'}
                raw_delta=left['generated_output_tokens_per_second']-right['generated_output_tokens_per_second']
                good_delta=left['SLO_qualified_output_tokens_per_second']-right['SLO_qualified_output_tokens_per_second']
                assert abs(raw_delta-sum(lost.values())-good_delta)<1e-8
                pairs.append(dict(start=index,raw_rate_delta=raw_delta,goodput_delta=good_delta,lost_rate_delta_by_failure=lost))
            if not pairs:continue
            identities.append(dict(family=main,left=label,right='StockToken',starts=pairs,
                start_mean_raw_rate_delta=float(np.mean([r['raw_rate_delta'] for r in pairs])),
                start_mean_goodput_delta=float(np.mean([r['goodput_delta'] for r in pairs])),
                start_mean_lost_rate_delta_by_failure={key:float(np.mean([r['lost_rate_delta_by_failure'][key] for r in pairs])) for key in pairs[0]['lost_rate_delta_by_failure']},
                evidence='EXACT_METRIC_ACCOUNTING_IDENTITY',scope='Goodput delta equals raw token-rate delta minus excluded token-rate deltas. The identity identifies metric losses,not runtime causes.'))
    # These associations describe the observed mixed trace. They are not online
    # prediction features or tests of independent causal effects.
    associations=[];conditions=[]
    for (family,name),rows in by_run.items():
        origin=min(json.loads(row['boundaries'])['arrival'] for row in rows)
        features=[]
        for row in rows:
            p=json.loads(row['prefill_queue_at_arrival']) if row.get('prefill_queue_at_arrival') else {}
            d=json.loads(row['decode_queue_at_arrival']) if row.get('decode_queue_at_arrival') else {}
            geometry=json.loads(row['prefill_first_query_geometry']) if row.get('prefill_first_query_geometry') else {}
            first_d=json.loads(row['decode_first_query_geometry']) if row.get('decode_first_query_geometry') else {}
            features.append(dict(arrival_seconds=json.loads(row['boundaries'])['arrival']-origin,
                input_tokens=float(row['input_tokens']),actual_output_tokens=float(row['output_tokens']),
                P_waiting_at_arrival=p.get('waiting_total'),P_running_at_arrival=p.get('running'),
                P_free_KV_blocks_at_arrival=p.get('free_KV_blocks'),D_waiting_at_arrival=d.get('waiting_total'),
                D_decode_ready_at_arrival=d.get('decode_ready_waiting'),D_active_decode_at_arrival=d.get('active_decode'),
                P_first_query_batch_requests=geometry.get('requests'),D_first_query_batch_requests=first_d.get('requests'),
                coordinator_wait_ms=float(row['Arrival_to_P_submit_ms']),
                P_accept_to_prepare_ms=float(row['P_accept_to_prepare_ms']),
                D_accept_to_prepare_ms=float(row['D_accept_to_prepare_ms'])))
        meta=dict(family=family,name=name,label=rows[0]['label'],start=int(rows[0]['start']))
        for feature in features[0]:
            for outcome in ('TTFT_ms','JOINT_PASS'):
                valid=[i for i,r in enumerate(features) if r[feature] is not None]
                values=[float(rows[i][outcome]) for i in valid]
                associations.append(dict(meta,feature=feature,outcome=outcome,
                    **descriptive_correlation([features[i][feature] for i in valid],values),
                    missing_requests=len(rows)-len(valid),evidence='DESCRIPTIVE_TRACE_ASSOCIATION',
                    scope='No p-value,independence claim or causal attribution. Actual output and realized batch are outcomes,not future-safe selector inputs.'))
        for condition in ('TTFT_PASS','JOINT_PASS'):
            for passed in (False,True):
                selected=[i for i,row in enumerate(rows) if row[condition]==passed]
                if not selected:continue
                conditions.append(dict(meta,condition=condition,passed=passed,requests=len(selected),
                    output_tokens=sum(int(rows[i]['output_tokens']) for i in selected),
                    feature_observed_requests={field:sum(features[i][field] is not None for i in selected) for field in features[0]},
                    feature_missing_requests={field:sum(features[i][field] is None for i in selected) for field in features[0]},
                    feature_means={field:float(np.mean([features[i][field] for i in selected if features[i][field] is not None]))
                        if any(features[i][field] is not None for i in selected) else None for field in features[0]},
                    evidence='DESCRIPTIVE_CONDITIONAL_TRACE',
                    scope='A scheduler snapshot excludes requests not yet admitted by the coordinator. Missing snapshots stay unknown.'))
    sources={str((ROOT/'analysis'/name).relative_to(REPO)):digest(ROOT/'analysis'/name) for name in (
        'request_failure_attribution.json','request_critical_paths.json')}
    value=dict(time=time.time(),passed=True,comparisons=comparisons,goodput_gap_identities=identities,
        frozen_trace_oracles=oracles,native_target_removals=native_targets,request_associations=associations,
        conditional_request_features=conditions,source_files_SHA256=sources,evidence='TRACE_ANALYSIS_AND_OFFLINE_METRIC_ORACLE',
        scope='Separate measured paired request paths,exact SLO accounting and non-implementable frozen-path removals. '
            'No host envelope is labelled exclusive GPU compute. The new controlled policy intervention remains independent evidence.')
    write(ROOT/'analysis/native_relative_causal_comparisons.json',value)
    for name,values in (('native_relative_paired_starts',comparisons),('request_trace_and_slo',joined),
            ('ttft_threshold_CDF',dist),('frozen_path_stage_oracles',oracles),
            ('frozen_native_target_stage_removals',native_targets),
            ('paired_native_stock_requests',paired_requests),('goodput_gap_accounting',identities),
            ('request_feature_associations',associations),('pass_fail_queue_features',conditions)):
        save_csv(ROOT/'figures'/(name+'.csv'),values)
    print('NATIVE_RELATIVE_CAUSAL_ANALYSIS_PASS',len(comparisons),'paired contrasts',len(oracles),'frozen-path points',flush=True)
    preserve();return value


if __name__=='__main__':analyze()
