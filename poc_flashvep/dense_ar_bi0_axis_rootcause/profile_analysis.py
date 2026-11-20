"""Join kernels by external IDs. Do not infer GPU time from host enclosures."""
import collections
import csv
import json
import numpy as np
from .common import ROOT,PREVIOUS,write,digest
from .queue_accounting import lines
from ..dense_ar_bi0_shortanswer_pd.cost_analysis import category,exclusive


def union(intervals):
    total=0;last=None
    for left,right in sorted(intervals):
        if right<=left:continue
        if last is None:last=[left,right]
        elif left<=last[1]:last[1]=max(last[1],right)
        else:total+=last[1]-last[0];last=[left,right]
    return total+(last[1]-last[0] if last is not None else 0)


def kernel_data(name,role='prefill'):
    letter='P' if role=='prefill' else 'D'
    folder=ROOT/'raw'/name;activity=folder/f'activity_{role}'
    counter=json.loads((activity/'activity_counters.json').read_text())
    meta=json.loads((folder/f'engine_step_external_ids_{role}.json').read_text())['rows']
    by_id={r['external_id']:r for r in meta}
    mapping={int(r['correlation']):int(r['external_id'])
        for r in csv.DictReader((activity/'kernels.csv.external.csv').open()) if int(r['external_kind'])==3}
    names={int(r['name_id']):r['kernel_name'] for r in csv.DictReader((activity/'kernels.csv.names.csv').open())}
    identified=set();identity_sources=[]
    for path in sorted((PREVIOUS/'analysis').glob('CostGrid_BI0_L4608_8064_Start*_v1_costs.json')):
        record=json.loads(path.read_text())
        identified.update(key for key,value in record.get('category_rules',{}).items() if value=='GEMM_API_identified')
        identity_sources.append(dict(path=str(path),SHA256=digest(path),evidence='HISTORICAL_KERNEL_NAME_IDENTITY'))
    stamps=json.loads((folder/f'activity_window_{role}.json').read_text())
    def host(point):return (point['host_lower_ns']+point['host_upper_ns'])/2e9
    h0,h1=host(stamps['left']),host(stamps['right'])
    c0,c1=stamps['left']['cupti_ns'],stamps['right']['cupti_ns']
    def mapped_host(ns):return h0+(ns-c0)*(h1-h0)/(c1-c0)
    bracket=max((stamps[s]['host_upper_ns']-stamps[s]['host_lower_ns'])/1e6 for s in ('left','right'))
    by_request=collections.defaultdict(list);copies=collections.defaultdict(list);all_roi=[];mapped_roi=[];roi_graphs=[]
    raw_queries=[]
    source_times={f'q{r["sample_id"]}':r for r in lines(folder/'requests.jsonl')}
    cfg=json.loads((folder/'configuration.json').read_text())
    selected=cfg.get('profile_sample_request_ids')
    if selected is not None:source_times={rid:r for rid,r in source_times.items() if rid in selected}
    target_steps={r['external_id'] for r in meta if any(q['request_id'] in source_times and
        (role=='prefill' or q['start']<source_times[q['request_id']]['input_tokens']) for q in r['queries'])}
    for row in meta:
        raw_queries.extend((q['request_id'],q['start'],q['count'],q['actual_batch'],q['padded_queries'])
                           for q in row['queries'] if q['request_id'].startswith('q'))
    excluded_graph_kernels=0
    for row in csv.DictReader((activity/'kernels.csv').open()):
        left,right=int(row['start_ns']),int(row['end_ns'])
        if right<=left:continue
        identifier=mapping.get(int(row['correlation']));step=by_id.get(identifier)
        ids={q['request_id'] for q in step['queries'] if q['request_id'].startswith('q')} if step else set()
        rid=next(iter(ids)) if len(ids)==1 else None
        if row['kind']=='kernel':
            if role=='decode' and cfg.get('runtime_only_correlation') and int(row['graph_id'])!=0:
                if c0<=left and right<=c1:excluded_graph_kernels+=1
                continue
            if c0<=left and right<=c1:
                all_roi.append((left,right))
                if rid:mapped_roi.append((left,right))
            if rid and identifier in target_steps:by_request[rid].append((left,right,category(names[int(row['name_id'])],identified)))
        elif rid and identifier in target_steps and row['kind'] in ('memcpy','memcpy2','memset'):
            copies[rid].append((left,right,int(row['bytes']),row['kind']))
        elif row['kind']=='graph' and left<c1 and right>c0:
            roi_graphs.append(dict(start=left,end=right,external_id=identifier))
    expected=json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']
    if selected is not None:expected=[q for q in expected if q['request_id'] in selected]
    expected=[(q['request_id'],q['start'],q['count'],q['requests'],q['padded_queries']) for q in expected]
    query_match=raw_queries==expected
    decisions={r['request_id']:r for r in lines(folder/'axis_decisions.jsonl')}
    handoffs={r['request_id']:r for r in lines(folder/'handoffs.jsonl')}
    submissions={r['request_id']:r for r in lines(folder/'request_submissions.jsonl') if r['role']=='decode'}
    rows=[]
    for rid in sorted(source_times,key=lambda s:int(s[1:])):
        kernels=by_request[rid];parts=exclusive(kernels)
        left=min((a for a,_,_ in kernels),default=None);right=max((b for _,b,_ in kernels),default=None)
        lower=decisions[rid]['time'] if role=='prefill' else submissions[rid]['time']
        upper=handoffs[rid]['source_ready'] if role=='prefill' else source_times[rid]['first_token']
        bounds=left is not None and mapped_host(left)>=lower-.0005 and mapped_host(right)<=upper+.0005
        operations=[(a,b) for a,b,_ in kernels]+[(a,b) for a,b,_,_ in copies[rid]]
        operation_union=union(operations)/1e6
        first_operation=min((a for a,_ in operations),default=None)
        last_operation=max((b for _,b in operations),default=None)
        pieces=None
        if first_operation is not None and role=='prefill':
            first_host,last_host=mapped_host(first_operation),mapped_host(last_operation)
            slope=(h1-h0)*1e9/(c1-c0)
            pieces=dict(dispatch_to_first_GPU_operation_ms=1000*(first_host-decisions[rid]['time']),
                GPU_operation_union_on_host_clock_ms=operation_union*slope,
                between_attributed_GPU_operation_gaps_on_host_clock_ms=((last_operation-first_operation)/1e6-operation_union)*slope,
                last_GPU_operation_to_parent_ready_ms=1000*(handoffs[rid]['received']-last_host),
                observed_P_dispatch_to_parent_ready_ms=1000*(handoffs[rid]['received']-decisions[rid]['time']))
            pieces['partition_residual_ms']=pieces['observed_P_dispatch_to_parent_ready_ms']-sum(v for k,v in pieces.items() if k!='observed_P_dispatch_to_parent_ready_ms')
            # CUPTI and host clocks are linked by the two measured stamp anchors.
            # GPU durations must use the same clock slope for this partition.
            pieces['clock_slope']=slope
        rows.append(dict(request_id=rid,prompt_id=source_times[rid]['prompt_id'],kernel_count=len(kernels),
            **{f'{letter}_kernel_union_ms':sum(parts.values())},named_categories_exclusive_ms=parts,
            first_attributed_GPU_kernel_host_time=mapped_host(left) if left is not None else None,
            last_attributed_GPU_kernel_host_time=mapped_host(right) if right is not None else None,
            arrival_to_first_attributed_GPU_kernel_ms=1000*(mapped_host(left)-source_times[rid]['arrival']) if left is not None else None,
            attributed_GPU_kernel_enclosure_ms=(right-left)/1e6 if left is not None else None,
            **{f'{letter}_copy_engine_union_ms':union([(a,b) for a,b,_,_ in copies[rid]])/1e6 if cfg.get('activity_copies') else None,
               f'{letter}_copy_record_bytes':sum(value for _,_,value,_ in copies[rid]) if cfg.get('activity_copies') else None},
            source_service_partition=pieces,
            source_host_bounds_passed=bool(bounds),
            scope='Kernel union excludes host gaps and copy-engine activity. Only API-correlated local copies enter the operation partition. '
                'The first attributed kernel can prepare inputs. It is not specifically the first GEMM. A kernel enclosure includes gaps and is not kernel-exclusive time. '
                'Remote-initiated DMA and uncorrelated operations can occupy its gaps. These gaps are not a physical-idle or CPU-exclusive measurement. '
                'Named categories are disclosed name-based classifications, not SM/HBM counters.'))
    coverage=union(mapped_roi)/union(all_roi) if all_roi else 0.
    graph_gate=not roi_graphs if role=='prefill' else all(r['external_id'] in by_id and r['external_id'] not in target_steps for r in roi_graphs)
    replay_witness=None
    if role=='decode' and cfg.get('runtime_only_correlation'):
        measured=[r for r in meta if any(q['request_id'].startswith('q') for q in r['queries'])]
        counted=sum(r.get('graph_replays',0) for r in measured)
        expected_replays=json.loads((folder/'engine_steps_decode.json').read_text())['graph_replays']
        actual_runtime=json.loads((folder/'actual_runtime_decode.json').read_text())
        replay_witness=dict(counted_replays=counted,worker_replays=expected_replays,
            no_replay_in_prompt_steps=all(not r.get('graph_replays') for r in measured if r['external_id'] in target_steps),
            decode_only_graph_policy='FULL_DECODE_ONLY' in actual_runtime['cudagraph_mode'])
        graph_gate=graph_gate and counted==expected_replays and replay_witness['no_replay_in_prompt_steps'] and replay_witness['decode_only_graph_policy']
    result=dict(name=name,rows=rows,query_spans_exact=query_match,ROI_kernel_time_mapped_fraction=coverage,
        unmapped_ROI_kernel_union_ms=(union(all_roi)-union(mapped_roi))/1e6,
        mean_unmapped_kernel_upper_bound_ms=(union(all_roi)-union(mapped_roi))/1e6/len(source_times),
        counters=counter,timestamp_bracket_max_ms=bracket,
        sampled_request_ids=selected,attributed_requests=len(source_times),
        copy_activity_measured=bool(cfg.get('activity_copies')),
        unqualified_graph_records_in_ROI=len(roi_graphs),
        excluded_decode_graph_kernel_records=excluded_graph_kernels,model_replay_phase_witness=replay_witness,
        graph_records_outside_measured_prefill_steps=graph_gate,
        category_identity_sources=identity_sources,
        attribution_gate=counter['passed'] and query_match and bracket<=.5 and coverage>=.99 and graph_gate and all(r['source_host_bounds_passed'] for r in rows),
        **{f'{letter}_kernel_union_mean_ms':float(np.mean([r[f'{letter}_kernel_union_ms'] for r in rows])),
           f'{letter}_named_category_means':{key:float(np.mean([r['named_categories_exclusive_ms'].get(key,0.) for r in rows]))
            for key in sorted({k for r in rows for k in r['named_categories_exclusive_ms']})}},
        evidence='DIAGNOSTIC_PENDING_OBSERVER_GATE',
        scope='Request attribution uses CUPTI API correlation IDs, not host containment. '
            'For D, only engine steps that compute original prompt queries enter suffix kernel costs. '
            'This is step-associated kernel time. Deferred sampling can belong to a later decode step. The definition is not the complete first-output GPU critical path. '
            'Any unqualified graph inside a measured prefill step fails the gate. Graphs in separate decode steps are not kernel-exclusive measurements. '
            'No D decode-exclusive time is inferred.')
    write(ROOT/'analysis'/(name+f'_{letter}_kernel_attribution.json'),result)
    return result


def triplet(names,role='prefill'):
    profile=kernel_data(names['profile'],role);values={};shapes={};outputs={}
    selected=profile.get('sampled_request_ids');subset_body={};subset_TTFT={}
    for label,name in names.items():
        folder=ROOT/'raw'/name
        metrics=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        timeline=json.loads((ROOT/'analysis'/(name+'_axis_timeline.json')).read_text())
        fields=('TTFT_mean_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms')
        values[label]={field:metrics[field] for field in fields}
        values[label]['P_prepare_to_ready_mean_ms']=timeline['summary']['P_prepare_to_ready_ms']['mean']
        if selected is not None:
            chosen=[r for r in timeline['rows'] if r['request_id'] in selected]
            assert len(chosen)==len(selected)
            subset_body[label]=float(np.mean([r['P_prepare_to_ready_ms'] for r in chosen]))
            subset_TTFT[label]=float(np.percentile([r['TTFT_ms'] for r in chosen],95))
        if role=='decode':values[label]['D_prepare_to_first_worker_mean_ms']=timeline['summary']['D_prepare_to_first_worker_output_ms']['mean']
        shapes[label]=[(q['request_id'],q['start'],q['count'],q['requests'],q['padded_queries']) for q in
            json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']]
        outputs[label]=[r['output_ids'] for r in sorted(lines(folder/'requests.jsonl'),key=lambda r:r['sample_id'])]
    changes={field:values['profile'][field]/np.mean([values[label][field] for label in ('clean_before','clean_after')])-1
        for field in values['profile']}
    observer=all(abs(float(v))<=.05 for v in changes.values())
    subset_change=None;subset_gate=True
    if selected is not None:
        subset_change=float(subset_body['profile']/np.mean([subset_body[v] for v in ('clean_before','clean_after')])-1)
        subset_gate=bool(abs(subset_change)<=.05)
        observer=bool(observer and subset_gate)
    matching=shapes['clean_before']==shapes['profile']==shapes['clean_after'] and outputs['clean_before']==outputs['profile']==outputs['clean_after']
    passed=observer and profile['attribution_gate'] and matching
    result=dict(names=names,relative_changes={key:float(v) for key,v in changes.items()},
        sampled_request_body_relative_change=subset_change,sampled_request_body_gate_passed=subset_gate,
        sampled_request_TTFT_p95_ms=subset_TTFT,
        observer_gate_passed=observer,matching_shapes_and_strict_outputs=matching,
        attribution_gate=profile['attribution_gate'],qualified=passed,kernel_measurement=profile,
        evidence='MEASURED_ONLINE' if passed else 'DIAGNOSTIC_OBSERVER_OR_ATTRIBUTION_GATE_FAILED',
        scope='One profiled start and two clean controls. The5% check uses point estimates, not an upper confidence bound. Failed timing is retained but not used as primary exclusive-GPU evidence.')
    write(ROOT/'analysis'/(names['profile']+'_observer_gate.json'),result)
    return result


def main():
    declarations=[]
    path=ROOT/'analysis/causal_P_profile_triplets.json'
    if path.exists():declarations.extend(json.loads(path.read_text())['results'])
    path=ROOT/'analysis/runtime_api_profile_triplets.json'
    if path.exists():declarations.extend(json.loads(path.read_text())['results'])
    path=ROOT/'analysis/sparse_P_profile_fallback_v1.json'
    if path.exists():
        declarations.extend(dict(axis=r['axis'],mode='NATURAL_SUBSAMPLE50',names=r['names'])
            for r in json.loads(path.read_text())['results'] if 'names' in r)
    for path in (ROOT/'analysis').glob('natural_Last4_*_profile_triplet.json'):
        declarations.append(json.loads(path.read_text()))
    rows=[]
    for row in declarations:
        if all((ROOT/'analysis'/(name+'_metrics.json')).exists() for name in row['names'].values()):
            rows.append(dict(axis=row.get('axis'),mode=row.get('mode','NATURAL_REPRODUCTION'),**triplet(row['names'])))
    write(ROOT/'analysis/P_kernel_profile_qualification_summary.json',dict(results=rows,
        scope='Qualified kernel evidence and failed diagnostics remain separate. No D kernel-exclusive claim is inferred.'))
    print(json.dumps([dict(axis=r['axis'],mode=r['mode'],qualified=r['qualified'],changes=r['relative_changes']) for r in rows],indent=2))


if __name__=='__main__':main()
