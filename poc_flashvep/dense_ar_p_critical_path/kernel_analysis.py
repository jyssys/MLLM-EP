"""Classify named kernels and retain overlap and attribution limits."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from .common import ROOT,write


def category(name):
    n=name.lower()
    if any(v in n for v in ('flashattn','flash_fwd','flash_attn','flashinfer','paged_attention')):return 'ATTENTION'
    if 'flash' in n and 'prepare_varlen' in n:return 'ATTENTION_METADATA'
    if any(v in n for v in ('nvjet','gemm','gemv','sgemm','bgemm')):return 'GEMM'
    if 'fused_add_rms_norm' in n:return 'FUSED_RMSNORM_RESIDUAL'
    if 'rms_norm' in n:return 'RMSNORM'
    if 'rotary' in n:return 'ROPE'
    if 'reshape_and_cache' in n:return 'KV_CACHE'
    if 'act_and_mul' in n or 'silu' in n:return 'ACTIVATION'
    if any(v in n for v in ('sample','top_k','topk','argmax','logit','rejection')):return 'SAMPLING_OR_LOGITS'
    if any(v in n for v in ('prepare_','slot_mapping','block_table','expand_idx','post_update','apply_write')):return 'METADATA'
    if any(v in n for v in ('copy','fillfunctor','gather')):return 'COPY_OR_INITIALIZE_KERNEL'
    if 'elementwise' in n:return 'OTHER_ELEMENTWISE'
    return 'UNKNOWN'


def merge(values):
    result=[]
    for a,b in sorted(values):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def duration(values):return sum(b-a for a,b in merge(values))


def intersect(values,left,right):
    # The caller can reuse a merged timeline for many request enclosures.
    import bisect
    ends=[b for a,b in values];index=bisect.bisect_right(ends,left);total=0
    for a,b in values[index:]:
        if a>=right:break
        total+=max(0,min(b,right)-max(a,left))
    return total


def analyze_activity(activity):
    roi=json.loads((activity/'actual_ROI.json').read_text());left=roi['left']['cupti_ns'];right=roi['right']['cupti_ns']
    counters=json.loads((activity/'activity_counters.json').read_text());assert counters['passed']
    names={int(r['name_id']):r['kernel_name'] for r in csv.DictReader((activity/'kernels.csv.names.csv').open())}
    records=[];groups=defaultdict(list);per_kernel=defaultdict(lambda:dict(count=0,summed_ns=0))
    for row in csv.DictReader((activity/'kernels.csv').open()):
        if row['kind']!='kernel':continue
        a,b=max(left,int(row['start_ns'])),min(right,int(row['end_ns']))
        if b<=a:continue
        kernel=names[int(row['name_id'])];kind=category(kernel);groups[kind].append([a,b]);records.append((a,b,kind))
        per_kernel[kernel]['count']+=1;per_kernel[kernel]['summed_ns']+=b-a;per_kernel[kernel]['category']=kind
    groups={key:merge(value) for key,value in groups.items()};all_busy=merge([[a,b] for a,b,c in records])
    edges=defaultdict(lambda:defaultdict(int))
    for kind,values in groups.items():
        for a,b in values:edges[a][kind]+=1;edges[b][kind]-=1
    active=defaultdict(int);partition=defaultdict(int);cursor=left
    for stamp,changes in sorted(edges.items()):
        kinds=tuple(sorted(k for k,v in active.items() if v>0))
        key=kinds[0] if len(kinds)==1 else 'CROSS_CATEGORY_OVERLAP' if kinds else 'NO_KERNEL'
        partition[key]+=stamp-cursor
        for kind,value in changes.items():active[kind]+=value
        cursor=stamp
    partition['NO_KERNEL']+=right-cursor
    assert sum(partition.values())==right-left
    return dict(ROI=roi,ROI_ns=right-left,kernel_count=len(records),kernel_busy_ns=duration(all_busy),
        kernel_busy_fraction=duration(all_busy)/(right-left),exclusive_time_partition_ns=dict(partition),
        category_union_ns={k:duration(v) for k,v in groups.items()},
        category_sum_minus_union_ns=sum(b-a for a,b,k in records)-duration(all_busy),
        per_kernel={k:v for k,v in sorted(per_kernel.items(),key=lambda kv:-kv[1]['summed_ns'])},
        intervals=all_busy,category_intervals=groups,counters=counters,
        scope='Kernel-name classification on actual CUDA activity. Unknown kernels remain UNKNOWN. '
            'Attention metadata is not the attention forward kernel. Fused Norm/residual is not divided. '
            'Union durations avoid double counting. Cross-category overlaps are separate. '
            'Projection subtypes cannot be recovered from a shared GEMM name. No GPU operator-to-request allocation is assumed.')


def cpu_test():
    assert category('FlashAttnFwdSm90 cutlass')=='ATTENTION'
    assert category('flash prepare_varlen_num_blocks_kernel')=='ATTENTION_METADATA'
    assert category('nvjet_tst_256x128')=='GEMM'
    assert category('fused_add_rms_norm_kernel')=='FUSED_RMSNORM_RESIDUAL'
    assert category('unknown_kernel')=='UNKNOWN'
    assert merge([[0,3],[2,4],[7,8]])==[[0,4],[7,8]]
    assert intersect([[0,4],[7,8]],2,7.5)==2.5


if __name__=='__main__':cpu_test();print('KERNEL_ACCOUNTING_CPU_TEST_PASS')
