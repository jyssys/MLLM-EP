"""Use paired server starts, not request counts, for performance intervals."""
import collections
import csv
import itertools
import json
import numpy as np
from .common import ROOT,write
from .figures import interval

CORE_RUNTIME=('common.py','serving.py','standard_worker.py','engine_options.py','ingress.py',
    'nixl_transport.py','partial_axis.py','variable_hidden.py')


def core_sources(name):
    hashes=json.loads((ROOT/'raw'/name/'source_hashes.json').read_text())
    prefix='poc_flashvep/dense_ar_bi0_shortanswer_pd/'
    return {file:hashes[prefix+file] for file in CORE_RUNTIME}


def actual_gpu_uuids(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    roles=('replica0','replica1') if cfg['method']=='Agg' else ('prefill','decode')
    return [str(json.loads((folder/f'actual_runtime_{role}.json').read_text())['uuid']).lower().removeprefix('gpu-') for role in roles]


def paired_ratio(numerator,denominator):
    a=np.asarray(numerator,dtype=float);b=np.asarray(denominator,dtype=float)
    assert len(a)==len(b) and len(a)>0
    point=float(a.mean()/b.mean()) if b.mean()>0 else None
    if len(a)<3:return dict(n=len(a),ratio=point,CI95=None)
    indices=list(itertools.product(range(len(a)),repeat=len(a)))
    ratios=[float(a[list(i)].mean()/b[list(i)].mean()) for i in indices if b[list(i)].mean()>0]
    bounds=[float(np.percentile(ratios,p)) for p in (2.5,97.5)] if len(ratios)==len(indices) else None
    differences=[float((a[list(i)]-b[list(i)]).mean()) for i in indices]
    return dict(n=len(a),ratio=point,CI95=bounds,
        difference=float((a-b).mean()),difference_CI95=[float(np.percentile(differences,p)) for p in (2.5,97.5)],
        scope='Ratio of paired-start means. Empirical bootstrap over independent starts. Not a request/token bootstrap.')


def main(fast=False):
    version='v2' if fast else 'v1';source=ROOT/'analysis'/f'BI0_shared_KV128_confirmation_{version}.json'
    if not source.exists():return None
    rows=json.loads(source.read_text())['results'];groups=collections.defaultdict(dict)
    for r in rows:groups[r['label']][r['seed']]=r
    summary={}
    for label,starts in groups.items():
        fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second',
                'SLO_qualified_output_tokens_per_second','Joint_SLO_attainment','output_tokens',
                'TTFT_mean_ms','mean_TBT_ms','E2E_p95_ms','completed_requests_per_second',
                'every_ITL_SLO_attainment','TPOT_SLO_attainment','decode_SLO_attainment','TTFT_SLO_attainment')
        summary[label]=dict(n=len(starts),cut=next(iter(starts.values()))['cut'],
            all_quality_CI_passed=all(r['metrics']['quality_gate']['CI_passed'] for r in starts.values()),
            global_max_ITL_ms=max(r['metrics']['max_ITL_ms'] for r in starts.values()),
            means={key:float(np.mean([r['metrics'][key] for r in starts.values()])) for key in fields},
            CI95={key:interval([r['metrics'][key] for r in starts.values()]) for key in fields})
    comparisons=[]
    inventory=json.loads((ROOT/'state/environment.json').read_text())['GPU_inventory']
    expected={int(row[0].strip()):row[1].strip().lower().removeprefix('gpu-') for row in list(csv.reader(inventory.splitlines()))[1:]}
    for numerator,denominator in (('Layer','ControlledNative'),('Token','ControlledNative'),
                                  ('Layer','ContinuousNative'),('Token','ContinuousNative'),('Layer','Token'),
                                  ('Layer','Agg'),('Token','Agg')):
        seeds=sorted(set(groups[numerator])&set(groups[denominator]))
        if not seeds:continue
        matched=True;runtime_matched=True;GPU_pair_matched=True;UUID_pair_matched=True
        for seed in seeds:
            configurations=[json.loads((ROOT/'raw'/groups[label][seed]['name']/'configuration.json').read_text()) for label in (numerator,denominator)]
            matched &= configurations[0]['arrival_offsets']==configurations[1]['arrival_offsets']
            matched &= configurations[0]['dataset_SHA256']==configurations[1]['dataset_SHA256']
            runtime_matched &= core_sources(groups[numerator][seed]['name'])==core_sources(groups[denominator][seed]['name'])
            GPU_pair_matched &= configurations[0]['gpus']==configurations[1]['gpus']==[0,1]
            UUID_pair_matched &= all(actual_gpu_uuids(groups[label][seed]['name'])==[expected[0],expected[1]] for label in (numerator,denominator))
        fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second','SLO_qualified_output_tokens_per_second')
        ratios={field:paired_ratio([groups[numerator][s]['metrics'][field] for s in seeds],
                    [groups[denominator][s]['metrics'][field] for s in seeds]) for field in fields}
        decode=all(ratios[f]['CI95'] is not None and ratios[f]['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms'))
        comparisons.append(dict(numerator=numerator,denominator=denominator,seeds=seeds,trace_and_dataset_exact=matched,
            core_runtime_source_exact=runtime_matched,declared_GPU_pair_exact=GPU_pair_matched,
            actual_GPU_UUID_pair_exact=UUID_pair_matched,
            core_runtime_files=list(CORE_RUNTIME),
            ratios=ratios,Native_relative_decode_5pct_gate=decode if 'Native' in denominator else None))
    result=dict(summary=summary,comparisons=comparisons,evidence='MEASURED_ONLINE',
        scope='Discovery-frozen static choices. Natural EOS; actual output totals can differ. '
        'Independent-start paired empirical intervals. Quality and decode safety are separate gates. '
        'C0 batch1 control is separate from C3 ordinary continuous Native/Agg. No sustainable capacity or dynamic gain.')
    write(ROOT/'analysis'/f'BI0_shared_KV128_confirmation_statistics_{version}.json',result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');main(parser.parse_args().fast)
