"""Attribute sequential serving kernels within observed request boundaries."""
import json
import time
import numpy as np
from .common import ROOT,bind,write,digest
from .critical_analysis import records,Timeline,csv_save
from .kernel_analysis import analyze_activity


def analyze():
    bind()
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from ..dense_ar_output_aware_v2.c3_analysis import means_ci
    from ..dense_ar_output_aware_v2.statistics import paired_difference,ratio
    plan=ROOT/'jobs/serial_serving_operator_cpc_v1.json';result=[];operators=[];request_rows=[]
    tests=json.loads(plan.read_text())['tests'];extra=ROOT/'jobs/serial_matched_work_operator_cpc_v1.json'
    if extra.exists():tests+=json.loads(extra.read_text())['tests']
    for test in tests:
        names=test['jobs'];folders=[ROOT/'raw'/name for name in names]
        if not all((folder/'dispatcher_result.json').exists() for folder in folders):
            result.append(dict(**test,evidence='NOT_RUN_OR_RUNNING'));continue
        if not all(json.loads((folder/'dispatcher_result.json').read_text())['complete'] for folder in folders):
            result.append(dict(**test,evidence='FAILED'));continue
        metrics=[summarize(name) for name in names];fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')
        ratios={key:metrics[1][key]/((metrics[0][key]+metrics[2][key])/2) for key in fields}
        outputs=[records(folder/'requests.jsonl') for folder in folders]
        work=[[(r['prompt_id'],r['output_tokens']) for r in sorted(rows,key=lambda r:r['sample_id'])] for rows in outputs]
        exact=all(rows==work[0] for rows in work)
        batch1=all(role['maximum_requests']==1 and role['maximum_queries']<=8192 for metric in metrics for role in metric['actual_geometry'].values())
        activity={role:analyze_activity(folders[1]/('activity_'+role)) for role in ('prefill','decode')}
        passed=exact and batch1 and all(abs(value-1)<=.05 for value in ratios.values())
        name=names[1];folder=folders[1];count=test['count'];rows=outputs[1]
        from ..dense_ar_output_aware_v2.work_contract import executed_prompt_work
        cfg=json.loads((folder/'configuration.json').read_text())
        work_audit=executed_prompt_work(cfg,folder,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
        handoff={r['request_id']:r for r in records(folder/'handoffs.jsonl')}
        submit={(r['request_id'],r['role']):r['time'] for r in records(folder/'request_submissions.jsonl')}
        observations=[]
        for role,value in activity.items():
            roi=value['ROI'];offset=roi['left']['cupti_ns']-(roi['left']['host_lower_ns']+roi['left']['host_upper_ns'])/2
            last_offset=roi['right']['cupti_ns']-(roi['right']['host_lower_ns']+roi['right']['host_upper_ns'])/2
            uncertainty=max(roi[key]['host_upper_ns']-roi[key]['host_lower_ns'] for key in ('left','right'))+abs(last_offset-offset)
            timelines={key:Timeline(spans) for key,spans in value['category_intervals'].items()}
            busy=Timeline(value['intervals'])
            for request in rows:
                rid='q'+str(request['sample_id']);events=request['token_events']
                intervals={'P_arrival_to_ready':(request['arrival'],handoff[rid]['source_ready'])} if role=='prefill' else {
                    'D_submit_to_first_worker':(submit[rid,'decode'],events[0]['worker_time']),
                    'D_subsequent_decode':(events[0]['worker_time'],events[-1]['worker_time'])}
                for phase,(left,right) in intervals.items():
                    a,b=left*1e9+offset,right*1e9+offset
                    assert roi['left']['cupti_ns']<=a<=b<=roi['right']['cupti_ns']
                    kernel_ms=busy.area(a,b)/1e6
                    observations.append(dict(name=name,label=test['label'],start=test['start'],family=test.get('family','SERIAL_BEST_FIXED_SERVICE_CONTROL'),request_id=rid,
                        prompt_id=request['prompt_id'],input_tokens=request['input_tokens'],output_tokens=request['output_tokens'],
                        role=role,phase=phase,host_enclosure_ms=1000*(right-left),GPU_kernel_union_ms=kernel_ms,
                        no_observed_kernel_ms=1000*(right-left)-kernel_ms,
                        category_union_ms={key:timeline.area(a,b)/1e6 for key,timeline in timelines.items()},
                        clock_mapping_uncertainty_ns=uncertainty,
                        scope='One request is alive. Named-kernel unions inside real host boundaries. '
                            'Categories can overlap. No-kernel time is not a DMA or physical-idle measurement.'))
            for category,ns in value['exclusive_time_partition_ns'].items():
                operators.append(dict(label=test['label'],start=test['start'],family=test.get('family','SERIAL_BEST_FIXED_SERVICE_CONTROL'),name=name,role=role,category=category,
                    ms_per_request=ns/1e6/count,evidence='ISOLATED_KERNEL_MEASUREMENT' if passed else 'DIAGNOSTIC'))
        request_rows.extend(observations)
        summary=[]
        for phase in sorted({r['phase'] for r in observations}):
            selected=[r for r in observations if r['phase']==phase]
            summary.append(dict(phase=phase,count=len(selected),
                mean_host_ms=float(np.mean([r['host_enclosure_ms'] for r in selected])),
                mean_GPU_kernel_union_ms=float(np.mean([r['GPU_kernel_union_ms'] for r in selected])),
                mean_no_observed_kernel_ms=float(np.mean([r['no_observed_kernel_ms'] for r in selected]))))
        result.append(dict(**test,evidence='ISOLATED_KERNEL_MEASUREMENT' if passed else 'DIAGNOSTIC',
            CPC_passed=passed,observer_ratios=ratios,exact_request_and_output_work=exact,actual_batch1_passed=batch1,
            metrics=metrics,phase_means=summary,actual_executed_work=work_audit,
            kernel_counters={role:value['counters'] for role,value in activity.items()}))
    comparisons=[]
    for family in dict.fromkeys(row['family'] for row in result):
        labels=('Layer8','HybridIso8Approx') if family=='SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL' else ('Layer','Hybrid')
        for label in labels:
            matching=[];work=[]
            for start in sorted({row['start'] for row in result if row['family']==family}):
                selected={row['label']:row for row in result if row['family']==family and row['start']==start and row.get('CPC_passed')}
                if label not in selected or 'StockToken' not in selected:continue
                left,right=selected[label],selected['StockToken'];a,b=[
                    {phase['phase']:phase for phase in row['phase_means']} for row in (left,right)]
                field={}
                for phase in ('P_arrival_to_ready','D_submit_to_first_worker','D_subsequent_decode'):
                    for key in ('mean_host_ms','mean_GPU_kernel_union_ms'):
                        field[phase+'_'+key]=[a[phase][key],b[phase][key]]
                field['observed_kernel_P_plus_D_first_ms']=[sum(phases[phase]['mean_GPU_kernel_union_ms']
                    for phase in ('P_arrival_to_ready','D_submit_to_first_worker')) for phases in (a,b)]
                matching.append(field)
                la,rt=[row['actual_executed_work']['nominal_static_partition'] for row in (left,right)]
                work.append(dict(start=start,left=la,right=rt,
                    P_rows_relative_error=(la['P_rows']-rt['P_rows'])/rt['P_rows'],
                    D_rows_relative_error=(la['D_rows']-rt['D_rows'])/rt['D_rows']))
            if not matching:continue
            comparisons.append(dict(family=family,left=label,right='StockToken',qualified_paired_starts=len(matching),work=work,
                fields={key:dict(delta_ms=paired_difference([row[key][0] for row in matching],[row[key][1] for row in matching]),
                    ratio=ratio([row[key][0] for row in matching],[row[key][1] for row in matching])) for key in matching[0]},
                scope='Qualified paired serial controls only. All failed CPC starts remain in results. '
                    'N1 has no server-start CI. Nominal iso-work has recorded integer/alignment error. '
                    'P+D-first kernel sum is not TTFT or suffix-only GPU time. It excludes subsequent decode and transfer/host waiting.'))
    value=dict(time=time.time(),plan_SHA256=digest(plan),results=result,comparisons=comparisons,
        scope='Serial serving Batch1 controls. These are not online Poisson TTFT comparisons or sustainable capacity. '
            'Original serving APIs execute Full36 Decode. No model-call observer is required for serial request attribution. '
            'The complete P/D activity includes metadata and sampling kernels. D-first is a suffix-plus-first-output enclosure. '
            'A failed CPC gate remains diagnostic. Keep these controls separate from natural C3 N3 results.')
    write(ROOT/'analysis/serial_serving_operator_analysis.json',value)
    csv_save(ROOT/'figures/serial_serving_operator_per_start.csv',result)
    csv_save(ROOT/'figures/serial_serving_operator_categories.csv',operators)
    csv_save(ROOT/'figures/serial_serving_request_phases.csv',request_rows)
    csv_save(ROOT/'figures/serial_serving_paired_service_comparisons.csv',comparisons)
    print('SERIAL_OPERATOR_ANALYSIS',[(r['label'],r['start'],r['evidence']) for r in result],flush=True)
    return value


if __name__=='__main__':analyze()
