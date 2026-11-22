"""Retain all observer triplets and distinguish GPU union from API overlap."""
import csv
import itertools
import json
import numpy as np
from .common import ROOT,write
from .metrics import summarize
from .profile_analysis import analyze
from .window_analysis import merge,duration,overlap

FIELDS=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')


def paired_ci(profile,clean):
    p=np.asarray(profile,float);c=np.asarray(clean,float);n=len(p)
    indices=np.asarray(list(itertools.product(range(n),repeat=n)))
    boot=p[indices].mean(axis=1)/c[indices].mean(axis=1)
    return dict(estimate=float(p.mean()/c.mean()),CI95=np.percentile(boot,[2.5,97.5]).tolist(),
        independent_triplets=n,resamples=len(indices),unit='Paired independent server-start triplet',
        qualifier='Pointwise empirical bootstrap. Small n limits confidence coverage.')


def api_gaps(name,role):
    path=ROOT/'raw'/name/f'activity_{role}/kernels.csv'
    kernels=[];api=[];matched={}
    for row in csv.DictReader(path.open()):
        a,b=int(row['start_ns'])/1e6,int(row['end_ns'])/1e6
        if row['kind']=='kernel':kernels.append((a,b))
        elif row['kind'] in ('driver','runtime','internal_api'):
            api.append((a,b));matched.setdefault(int(row['correlation']),[]).append((a,b,row['kind'],row['api_id']))
    busy=merge(kernels);gaps=[(busy[i][1],busy[i+1][0]) for i in range(len(busy)-1)]
    return dict(interior_GPU_kernel_gap_union_ms=duration(gaps),
        gap_overlap_recorded_CUDA_API_ms=duration(overlap(gaps,api)),
        CUDA_API_union_ms=duration(api),recorded_CUDA_API_count=len(api),
        GPU_kernel_count=len(kernels),
        scope='CUPTI timestamps measure kernel gaps and temporal API overlap. '
            'API duration is host time. It is not added to GPU time. '
            'A gap outside an API can include host preparation or stream waits. Its cause is not identified by subtraction. '
            'Only v3 captures with the explicit API-export collector support this analysis.')


def main():
    contract=json.loads((ROOT/'configs/operator_repeat_observer_contract_v3.json').read_text())
    cells=[]
    for bucket,cut in ((0,2),(0,4),(3,4)):
        for axis in ('Layer','Token'):
            tests=[t for t in contract['tests'] if (t['input_bucket'],t['cut'],t['axis'])==(bucket,cut,axis)]
            assert len(tests)==3
            triplets=[]
            for test in tests:
                names=test['jobs']
                assert all(json.loads((ROOT/'raw'/n/'dispatcher_result.json').read_text())['complete'] for n in names)
                metric=[summarize(n) for n in names]
                reference={f:(metric[0][f]+metric[2][f])/2 for f in FIELDS}
                individual={f:metric[1][f]/reference[f] for f in FIELDS}
                stability={f:metric[0][f]/metric[2][f] for f in FIELDS}
                roles={}
                for role in ('prefill','decode'):
                    cache=ROOT/'analysis'/(names[1]+'_'+role+'_kernel_decomposition.json')
                    decomp=json.loads(cache.read_text()) if cache.exists() else analyze(names[1],role)
                    counters=json.loads((ROOT/'raw'/names[1]/f'activity_{role}/activity_counters.json').read_text())
                    assert counters['passed']
                    # Sum only the prefill/continuation frames. Decode graphs stay separate.
                    phases={k:v for k,v in decomp['phase_summary'].items() if k in
                        ('P_PREFIX_PREFILL','PREFIX_UPPER_CONTINUATION','SUFFIX_FULL36_CONTINUATION')}
                    categories={}
                    for phase in phases.values():
                        for key,value in phase['tagged_operator_kernel_union_ms'].items():categories[key]=categories.get(key,0)+value
                    n=metric[1]['n']
                    roles[role]=dict(complete_kernel_attribution=decomp['mapped_kernel_fraction']==1,
                        all_kernel_records=decomp['kernel_records'],unmapped_kernels=decomp['unmapped_kernels'],
                        prompt_GPU_kernel_union_ms_per_request=sum(v['total_GPU_kernel_union_ms'] for v in phases.values())/n,
                        prompt_operator_union_ms_per_request={k:v/n for k,v in categories.items()},
                        phases=phases,API_gap_overlap=api_gaps(names[1],role))
                triplets.append(dict(start=test['start'],jobs=names,profile=metric[1],mean_clean=reference,
                    profile_over_mean_clean=individual,clean1_over_clean2=stability,
                    individual_5percent_pass=all(.95<=r<=1.05 for r in individual.values()),roles=roles))
            cis={f:paired_ci([t['profile'][f] for t in triplets],[t['mean_clean'][f] for t in triplets]) for f in FIELDS}
            passed=all(.95<=v['CI95'][0]<=v['CI95'][1]<=1.05 for v in cis.values())
            passed=passed and all(r['complete_kernel_attribution'] for t in triplets for r in t['roles'].values())
            cell=dict(input_bucket=bucket,cut=cut,axis=axis,paired_triplet_CI=cis,triplets=triplets,
                observer_gate_passed=passed,evidence='ISOLATED_KERNEL_MEASUREMENT' if passed else 'DIAGNOSTIC',
                scope='Three frozen independent warm-shape triplets. Actual serving operators. '
                    'Warmup uses prompt-shape foreknowledge. These are isolated calibration results,not online serving speedups. '
                    'Every individual observer failure remains in the table. No repeat-until-pass selection.')
            cells.append(cell)
            print(bucket,cut,axis,'PASS' if passed else 'DIAGNOSTIC',
                {f:[round(v,4) for v in c['CI95']] for f,c in cis.items()},flush=True)
    write(ROOT/'analysis/operator_repeat_observer_analysis_v3.json',dict(cells=cells,
        scope='GPU operator unions and host API overlap retain separate units and scopes. SM utilization and HBM traffic are NOT_MEASURED.'))


if __name__=='__main__':main()
