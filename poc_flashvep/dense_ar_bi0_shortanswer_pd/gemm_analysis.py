"""Report projection-only service separately from complete model service."""
import csv
import itertools
import json
import numpy as np
from .common import ROOT,write
from .profile_results import union


def analyze(name):
    folder=ROOT/'raw'/name;data=json.loads((folder/'gemm_API_costs.json').read_text())
    activity=folder/'activity_all';counters=json.loads((activity/'activity_counters.json').read_text())
    kernels=[(int(r['start_ns']),int(r['end_ns'])) for r in csv.DictReader((activity/'kernels.csv').open()) if r['kind']=='kernel']
    rows=[]
    for row in data['rows']:
        values={label:[s['CUDA_enclosure_ms'] for s in part if not s['warmup']] for label,part in row['samples'].items()}
        clean=values['clean_before']+values['clean_after']
        changes={f'p{p}':float(np.percentile(values['profile'],p)/np.mean([np.percentile(values[label],p) for label in ('clean_before','clean_after')])-1) for p in (50,95)}
        measured=[]
        for sample in row['samples']['profile']:
            if sample['warmup']:continue
            left=sample['CUPTI_left']['cupti_ns'];right=sample['CUPTI_right']['cupti_ns']
            measured.append(union([(max(a,left),min(b,right)) for a,b in kernels if a<right and b>left])/1e6)
        gate=counters['passed'] and all(v>0 for v in measured) and all(abs(v)<=.05 for v in changes.values())
        flops=sum(2*r['M']*r['N']*r['K'] for r in row['GEMM_shapes'])
        parameter_bytes=sum(2*r['N']*r['K'] for r in row['GEMM_shapes'])
        rows.append(dict(input_tokens=row['input_tokens'],cut=row['cut'],axis=row['axis'],
            actual_linear_calls=row['actual_linear_calls'],logical_rows=row['logical_rows'],structural_GEMM_FLOPs=flops,
            structural_BF16_projection_parameter_bytes=parameter_bytes,
            FLOPs_per_parameter_byte=flops/parameter_bytes,
            parameter_scope='Sum of distinct projection matrix sizes used once per forward. Not measured HBM reads, bandwidth or cache misses.',
            clean_graph_CUDA_enclosure_p50_ms=float(np.percentile(clean,50)),
            clean_graph_CUDA_enclosure_p95_ms=float(np.percentile(clean,95)),
            kernel_union_mean_ms=float(np.mean(measured)),observer_gate_passed=gate,observer_relative_change=changes,
            event_based_TFLOPs_per_second=flops/(np.percentile(clean,50)*1e9),
            evidence='ISOLATED_KERNEL_MEASUREMENT' if gate else 'DIAGNOSTIC_OBSERVER_GATE_FAILED',
            scope='Projection-only CUDA graph through actual serving modules. Event enclosure is distinct from CUPTI kernel union. '
            'Arithmetic FLOPs are structural. This is not the complete forward, TTFT or measured HBM bandwidth.'))
    write(ROOT/'analysis'/(name+'_GEMM_metrics.json'),dict(rows=rows,counters=counters))
    return rows


def main():
    starts=[]
    for folder in sorted((ROOT/'raw').glob('GEMM_API_BI0_Start*_v1')):
        if (folder/'gemm_API_costs.json').exists():starts.append(analyze(folder.name))
    pairs=[]
    for length in (4608,8064):
        for cut in (1,2,4,6,8,12):
            entries=[{r['axis']:r for r in start if r['input_tokens']==length and r['cut']==cut} for start in starts]
            entries=[r for r in entries if len(r)==2]
            if not entries:continue
            assert all(r['Layer']['structural_GEMM_FLOPs']==r['Token']['structural_GEMM_FLOPs'] for r in entries)
            a=np.array([r['Layer']['clean_graph_CUDA_enclosure_p50_ms'] for r in entries])
            b=np.array([r['Token']['clean_graph_CUDA_enclosure_p50_ms'] for r in entries])
            draws=[float(b[list(i)].mean()/a[list(i)].mean()) for i in itertools.product(range(len(a)),repeat=len(a))]
            pairs.append(dict(input_tokens=length,cut=cut,Token_suffix=length*cut//36,independent_starts=len(a),
                Layer_GEMM_graph_p50_ms=float(a.mean()),Token_GEMM_graph_p50_ms=float(b.mean()),
                Token_over_Layer=float(b.mean()/a.mean()),paired_start_empirical_CI95=[float(np.percentile(draws,p)) for p in (2.5,97.5)] if len(a)>=3 else None,
                qualified_kernel_attribution=all(r[axis]['observer_gate_passed'] for r in entries for axis in ('Layer','Token')),
                scope='Mean of independent-start p50 values. Same logical rows and projection FLOPs. Not full-forward or online gain.'))
    write(ROOT/'analysis/BI0_GEMM_API_summary.json',dict(pairs=pairs,evidence='ISOLATED_GEMM_API_MEASUREMENT'))
    print(json.dumps(pairs,indent=2))


if __name__=='__main__':main()
