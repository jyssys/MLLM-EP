"""Use kernel unions only after the clean/profile/clean observer check."""
import csv
import collections
import json
import numpy as np
from .common import ROOT,write


def union(intervals):
    end=None;total=0
    for left,right in sorted(intervals):
        if end is None or left>end:total+=right-left;end=right
        elif right>end:total+=right-end;end=right
    return total


def main():
    results={}
    for bi in (False,True):
        groups={}
        for label in ('clean_before','profile','clean_after'):
            name=f'decode8192_BI{int(bi)}_{label}_C1_O16_v2';folder=ROOT/'raw'/name
            if not (folder/'single_gpu_results.json').exists():continue
            rows=[r for r in json.loads((folder/'single_gpu_results.json').read_text())['rows'] if not r['warmup']]
            groups[label]=dict(name=name,folder=folder,rows=rows,
                mean_TBT_ms=float(np.mean([r['mean_TBT_ms'] for r in rows])),
                TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in rows])),
                mean_decode_ROI_wall_ms=float(np.mean([r['legacy_prepared_decode_end_minus_first_ms'] for r in rows])))
        if len(groups)!=3:continue
        baseline={key:float(np.mean([groups[label][key] for label in ('clean_before','clean_after')]))
            for key in ('mean_TBT_ms','TTFT_mean_ms','mean_decode_ROI_wall_ms')}
        profile=groups['profile'];relative={key:profile[key]/baseline[key]-1 for key in baseline}
        gate=all(abs(value)<=.05 for value in relative.values())
        counters=json.loads((profile['folder']/'activity_counters.json').read_text())
        kernels=[(int(r['start_ns']),int(r['end_ns']),int(r['name_id']),int(r['graph_node']))
            for r in csv.DictReader((profile['folder']/'kernels.csv').open()) if r['kind']=='kernel']
        samples=[]
        totals=collections.Counter();counts=collections.Counter()
        for row in profile['rows']:
            roi=row['decode_kernel_ROI'];left=roi['left']['cupti_ns'];right=roi['right']['cupti_ns']
            selected=[(max(a,left),min(b,right),name,graph) for a,b,name,graph in kernels if a<right and b>left]
            for a,b,name,_ in selected:totals[name]+=b-a;counts[name]+=1
            busy=union([(a,b) for a,b,_,_ in selected]);duration=right-left
            enclosures=row['CUDA_step_enclosures_ms']
            samples.append(dict(prompt_id=row['prompt_id'],ROI_ms=duration/1e6,kernel_union_ms=busy/1e6,
                kernel_union_per_15_decode_steps_ms=busy/15e6,kernel_inactivity_ms=(duration-busy)/1e6,
                kernel_count=len(selected),graph_kernel_count=sum(bool(x[3]) for x in selected),
                decode_step_GPU_enclosures_ms=enclosures[1:],
                engine_step_wall_ms=[r['wall_ms'] for r in row['engine_steps'][1:]],
                interstep_Host_gap_ms=[r['prior_host_gap_ms'] for r in row['engine_steps'][1:]],
                client_mean_TBT_ms=row['mean_TBT_ms']))
        names={int(r['name_id']):r['kernel_name'] for r in csv.DictReader((profile['folder']/'kernels.csv.names.csv').open())}
        operator_rows=[dict(kernel_name=names[index],kernel_calls=counts[index],
            duration_sum_ms_per_decode_step=value/(1e6*15*len(samples))) for index,value in totals.most_common()]
        results[f'BI{int(bi)}']=dict(observer_gate_passed=gate and counters['passed'],relative_change=relative,
            baseline=baseline,profile={k:v for k,v in profile.items() if k not in ('folder','rows')},
            activity_counters=counters,samples=samples,
            per_kernel_name=operator_rows,operator_scope='Individual kernel duration sums, not exclusive category attribution. Categories can overlap across streams. Ninety observed decode steps belong to one profiled server start.',
            GPU_exclusive_decode_per_step_mean_ms=float(np.mean([r['kernel_union_per_15_decode_steps_ms'] for r in samples])),
            evidence='MEASURED_DIAGNOSTIC' if gate and counters['passed'] else 'DIAGNOSTIC_OBSERVER_GATE_FAILED',
            scope='Kernel union within synchronized first-sample-to-final-output ROI. Fifteen decode steps. Host gaps are reported separately. One profiled start and two clean starts per numerical path. Not stock Main throughput.')
    write(ROOT/'analysis/decode_kernel_host_breakdown_v2.json',dict(results=results,
        scope='Do not interpret a failed observer arm as clean Primary kernel timing. No finite-cohort or sustainable capacity claim.'))
    print(json.dumps({key:{k:v for k,v in row.items() if k not in ('samples','per_kernel_name')} for key,row in results.items()},indent=2))


if __name__=='__main__':main()
