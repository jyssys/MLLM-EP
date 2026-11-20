"""Keep event enclosures, kernel unions and online gains distinct."""
import collections
import csv
import itertools
import json
import re
import numpy as np
from .common import ROOT,write


def category(name,identified_GEMM=()):
    # FlashAttention also uses CUTLASS. Never label every CUTLASS kernel GEMM.
    if re.search(r'flash_fwd|flash_attn|FlashAttn|fmha|paged_attention',name,re.I):return 'Attention_named'
    if name in identified_GEMM:return 'GEMM_API_identified'
    if re.search(r'gemm|cublas|matmul|xmma',name,re.I):return 'GEMM_named'
    if re.search(r'rms_norm|layer_norm|rmsnorm',name,re.I):return 'Normalization_named'
    if re.search(r'rotary|reshape_and_cache|cache_kv',name,re.I):return 'KV_RoPE_named'
    return 'Other_unclassified'


def exclusive(intervals):
    events=[]
    for a,b,key in intervals:
        if b>a:events.extend([(a,key,1),(b,key,-1)])
    active=collections.Counter();result=collections.Counter();last=None
    for stamp,key,delta in sorted(events):
        if last is not None and stamp>last:
            keys=[k for k,v in active.items() if v>0]
            if keys:result[keys[0] if len(keys)==1 else 'Cross_category_overlap']+=stamp-last
        active[key]+=delta;last=stamp
    return {key:value/1e6 for key,value in result.items()}


def analyze(name,output_suffix='_costs.json'):
    folder=ROOT/'raw'/name;source=json.loads((folder/'isolated_axis_costs.json').read_text())
    groups=collections.defaultdict(dict)
    for row in source['rows']:groups[(row['input_tokens'],row['cut'],row['axis'])][row['pass_label']]=row
    activity=folder/'activity_all'
    counters=json.loads((activity/'activity_counters.json').read_text())
    names={int(r['name_id']):r['kernel_name'] for r in csv.DictReader((activity/'kernels.csv.names.csv').open())}
    raw_kernels=[(int(r['start_ns']),int(r['end_ns']),names[int(r['name_id'])])
                 for r in csv.DictReader((activity/'kernels.csv').open()) if r['kind']=='kernel']
    identified=set();identities=[]
    for primitive in source.get('primitive_identity',[]):
        left=primitive['CUPTI_left']['cupti_ns'];right=primitive['CUPTI_right']['cupti_ns']
        observed=sorted({name for a,b,name in raw_kernels if a<right and b>left})
        identified.update(observed);identities.append(dict(primitive,kernel_names=observed))
    kernels=[(a,b,category(name,identified)) for a,b,name in raw_kernels]
    output=[]
    for (length,cut,axis),passes in groups.items():
        values={label:[s['CUDA_enclosure_ms'] for s in row['samples'] if not s['warmup']] for label,row in passes.items()}
        change={}
        for p in (50,95):
            clean=np.mean([np.percentile(values[label],p) for label in ('clean_before','clean_after')])
            change[f'p{p}']=float(np.percentile(values['profile'],p)/clean-1)
        gate=counters['passed'] and all(abs(v)<=.05 for v in change.values())
        measured=[]
        for sample in passes['profile']['samples']:
            if sample['warmup']:continue
            left=sample['CUPTI_left']['cupti_ns'];right=sample['CUPTI_right']['cupti_ns']
            selected=[(max(a,left),min(b,right),key) for a,b,key in kernels if a<right and b>left]
            parts=exclusive(selected)
            measured.append(dict(kernel_union_ms=sum(parts.values()),categories_exclusive_ms=parts))
        clean=values['clean_before']+values['clean_after']
        output.append(dict(input_tokens=length,cut=cut,axis=axis,actual_batch=1,
            valid_GEMM_rows=length*cut,actual_queries=passes['profile']['actual_queries'],
            valid_queries=passes['profile']['valid_queries'],strict_hidden_equal=all(r['strict_hidden_equal'] for r in passes.values()),
            clean_CUDA_enclosure_ms={f'p{p}':float(np.percentile(clean,p)) for p in (50,95,99)},
            observer_relative_change=change,observer_gate_passed=gate,
            kernel_union_ms=float(np.mean([r['kernel_union_ms'] for r in measured])),
            named_categories_exclusive_ms={key:float(np.mean([r['categories_exclusive_ms'].get(key,0.) for r in measured]))
                for key in sorted({k for r in measured for k in r['categories_exclusive_ms']})},
            evidence='ISOLATED_KERNEL_MEASUREMENT' if gate else 'DIAGNOSTIC_OBSERVER_GATE_FAILED',
            scope='Event enclosure is not kernel-exclusive time. Named categories use disclosed regex rules. '
            'Cross-category overlap is a separate bucket. This excludes transfer, queues and online scheduling.'))
    write(ROOT/'analysis'/(name+output_suffix),dict(name=name,rows=output,independent_server_starts=1,
        category_rules={k:category(k,identified) for k in names.values()},
        serving_Linear_API_kernel_identity=identities,counters=counters))
    return output


def main():
    starts=[]
    for folder in sorted((ROOT/'raw').glob('CostGrid_BI0_L4608_8064_Start*_v1')):
        if (folder/'isolated_axis_costs.json').exists():starts.append(dict(name=folder.name,rows=analyze(folder.name)))
    pairs=[]
    for length in (4608,8064):
        for cut in (1,2,4,6,8,12):
            values=[]
            for start in starts:
                selected={r['axis']:r for r in start['rows'] if r['input_tokens']==length and r['cut']==cut}
                if len(selected)==2:values.append(selected)
            if not values:continue
            a=np.array([v['Layer']['clean_CUDA_enclosure_ms']['p50'] for v in values])
            b=np.array([v['Token']['clean_CUDA_enclosure_ms']['p50'] for v in values])
            draws=[float(b[list(i)].mean()/a[list(i)].mean()) for i in itertools.product(range(len(values)),repeat=len(values))]
            pairs.append(dict(input_tokens=length,cut=cut,token_suffix=length*cut//36,starts=len(values),
                Layer_CUDA_enclosure_p50_ms=float(a.mean()),Token_CUDA_enclosure_p50_ms=float(b.mean()),
                Token_over_Layer=float(b.mean()/a.mean()),paired_start_empirical_CI95=[float(np.percentile(draws,p)) for p in (2.5,97.5)] if len(values)>=3 else None,
                category_kernel_gates_passed=all(v[axis]['observer_gate_passed'] for v in values for axis in ('Layer','Token')),
                scope='Ratio of start-level event p50 means. Empirical start bootstrap. Not TTFT or sustainable capacity.'))
    write(ROOT/'analysis/BI0_isolated_axis_cost_summary.json',dict(starts=[s['name'] for s in starts],pairs=pairs,
        evidence='ISOLATED_FORWARD_MEASUREMENT',scope='Clean event costs remain separate from observer-qualified kernel attribution.'))
    print(json.dumps(pairs,indent=2))


if __name__=='__main__':main()
