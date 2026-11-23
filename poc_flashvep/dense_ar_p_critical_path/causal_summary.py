"""Summarize qualified P activity without converting waiting into own compute."""
from collections import defaultdict
import json
import time
import numpy as np
from .common import ROOT,bind,write,digest
from .critical_analysis import csv_save


def bucket_input(length):
    for label,a,b in (('3K-6K',3000,6000),('6K-10K',6000,10000),('10K-20K',10000,20000),('20K-36K',20000,36001)):
        if a<=length<b:return label
    return 'OUTSIDE_DECLARED_BUCKETS'


def bucket_output(length):
    return '1-16' if length<=16 else '17-64' if length<=64 else '65-128' if length<=128 else '129+'


def analyze():
    bind()
    from ..dense_ar_output_aware_v2.statistics import paired_difference,ratio
    from ..dense_ar_output_aware_v2.c3_analysis import means_ci
    source=ROOT/'analysis/p_only_fulltrace_cpc_analysis.json';result=json.loads(source.read_text())
    observations=[];per_request=[];primary=defaultdict(dict)
    for item in result['results']:
        if 'detail' not in item:continue
        primary_family=item.get('family')=='PUBLISHED_TRACE_REPRODUCTION_SAME_PAIR'
        detail=json.loads((ROOT/item['detail']).read_text());kernels=detail['kernels'];path=detail['path']
        name=item['jobs'][1];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());count=cfg['count']
        row=dict(name=name,label=item['label'],start=item['start'],family=item.get('family','SEPARATE_STOCK_CALIBRATION'),
            evidence=item['evidence'],observer_passed=item['observer_gate']['passed'],requests=count,
            GPU_pair=cfg['gpus'],seed=cfg['seed'],B=8192,S=1024,
            P_kernel_resource_demand_ms_per_request=kernels['kernel_busy_ns']/1e6/count,
            P_kernel_busy_fraction=kernels['kernel_busy_fraction'],
            Arrival_to_P_ready_mean_ms=path['means'].get('Arrival_to_P_ready_ms'),
            P_kernel_overlapping_path_mean_ms=path['means'].get('all_P_kernel_inside_arrival_to_ready_ms'),
            no_observed_P_kernel_path_mean_ms=path['means'].get('no_observed_P_kernel_ms'),
            own_single_request_forward_kernel_mean_ms=path['means'].get('own_single_request_forward_kernel_ms'),
            shared_forward_kernel_mean_ms=path['means'].get('shared_forward_kernel_ms'),
            clock_mapping_uncertainty_ns=path.get('mapping_uncertainty_bound_ns'),
            source_window_full_fraction=path['credit'].get('full_fraction'),
            source_window_joint_arrival_to_dispatch_ms=path['credit'].get('mean_arrival_to_dispatch_full_window_overlap_ms'),
            source_window_pending_segment_status=path['credit'].get('pending_segment_status','UNBOUNDED'),
            category_resource_demand_ms_per_request={key:value/1e6/count for key,value in item['kernel_partition_ns'].items()
                if key!='NO_KERNEL'},
            scope='Resource demand is global P kernel union divided by request count. '
                'Kernel overlap along a request path includes preceding and shared requests. These are different denominators. '
                'No observed P kernel is not all-engine physical idle. DMA and HBM are not measured.')
        observations.append(row)
        if primary_family:primary[item['start']][item['label']]=row
        for request in path['rows']:
            if 'input_tokens' not in request:
                # Stock has no qualified per-request model observer. Its host table still identifies requests.
                from .critical_analysis import records
                outputs={'q'+str(r['sample_id']):r for r in records(ROOT/'raw'/name/'requests.jsonl')}
                value=outputs[request['request_id']]
                request=dict(request,input_tokens=value['input_tokens'],output_tokens=value['output_tokens'],prompt_id=value['prompt_id'])
            if primary_family:
                per_request.append(dict(request,name=name,label=item['label'],start=item['start'],evidence=item['evidence'],
                    observer_passed=item['observer_gate']['passed'],input_bucket=bucket_input(request['input_tokens']),
                    output_bucket=bucket_output(request['output_tokens'])))
    comparisons=[]
    for left,right in (('Layer','StockToken'),('Hybrid','StockToken'),('Hybrid','Layer')):
        present=[(values[left],values[right]) for _,values in sorted(primary.items()) if left in values and right in values]
        qualified=[(a,b) for a,b in present if a['observer_passed'] and b['observer_passed']]
        fields=('P_kernel_resource_demand_ms_per_request','Arrival_to_P_ready_mean_ms',
            'P_kernel_overlapping_path_mean_ms','no_observed_P_kernel_path_mean_ms')
        tests={key:dict(delta_ms=paired_difference([a[key] for a,b in qualified],[b[key] for a,b in qualified]),
            ratio=ratio([a[key] for a,b in qualified],[b[key] for a,b in qualified])) for key in fields} if qualified else {}
        categories=sorted(set(key for a,b in qualified for value in (a,b) for key in value['category_resource_demand_ms_per_request']))
        operators={key:paired_difference([a['category_resource_demand_ms_per_request'].get(key,0) for a,b in qualified],
            [b['category_resource_demand_ms_per_request'].get(key,0) for a,b in qualified]) for key in categories} if qualified else {}
        comparisons.append(dict(left=left,right=right,present_paired_starts=len(present),qualified_paired_starts=len(qualified),
            all_three_predeclared_pairs_qualified=len(present)==3 and len(qualified)==3,tests=tests,
            operator_resource_demand_delta_ms_per_request=operators,
            scope='All outcomes remain in the observation table. These summaries use only observer-qualified paired captures. '
                'Fewer than three qualified pairs do not establish an N3 result. These are pointwise empirical intervals. '
                'Operator differences are best-vs-best,not an attention-only intervention or exact iso-work comparison.'))
    groups=defaultdict(list)
    for row in per_request:groups[(row['label'],row['start'],row['input_bucket'],row['output_bucket'],row['evidence'])].append(row)
    buckets=[]
    for (label,index,inp,out,evidence),rows in groups.items():
        fields=('Arrival_to_P_ready_ms','all_P_kernel_inside_arrival_to_ready_ms','no_observed_P_kernel_ms')
        buckets.append(dict(label=label,start=index,input_bucket=inp,output_bucket=out,evidence=evidence,count=len(rows),
            **{key:float(np.mean([r[key] for r in rows])) for key in fields},
            scope='Descriptive mixed-replay subgroup. Not an intervention on output length. No request-level superiority CI.'))
    value=dict(time=time.time(),source_SHA256=digest(source),observations=observations,comparisons=comparisons,
        descriptive_input_output_buckets=buckets,
        scope='Historical1.76-second host difference is not retroactively decomposed with this new capture. '
            'This is a new same-pair,three-trace attribution study. Keep separate Stock calibration out of paired comparisons. '
            'Attention pairs,HBM,SM utilization and DMA are not inferred from kernel names. '
            'No per-request Stock forward attribution is available after its failed strict observer qualification.')
    write(ROOT/'analysis/p_critical_causal_summary.json',value)
    csv_save(ROOT/'figures/p_kernel_resource_and_path_per_start.csv',observations)
    csv_save(ROOT/'figures/p_kernel_resource_and_path_paired.csv',comparisons)
    csv_save(ROOT/'figures/p_input_output_descriptive_buckets.csv',buckets)
    print('P_CAUSAL_SUMMARY',len(observations),[(v['left'],v['right'],v['qualified_paired_starts']) for v in comparisons],flush=True)
    return value


if __name__=='__main__':analyze()
