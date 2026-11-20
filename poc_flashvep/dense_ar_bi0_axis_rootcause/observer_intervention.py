"""Use failed profiling only as a disclosed source-overhead diagnostic."""
import json
import numpy as np
from .common import ROOT,write


def main():
    declaration=ROOT/'configs/sparse_profile_additional_body_gate_v1.json'
    if not declaration.exists():
        write(declaration,dict(scope='Additional stricter check before sampled profiles execute. Keep the existing full200 observer gate. '
            'The sampled50 P-prepare-to-ready mean must also change by no more than5%. Global averaging must not hide a larger sampled-request service perturbation.',
            sampled_body_limit=.05))
    source=ROOT/'analysis/runtime_api_profile_triplets.json'
    if not source.exists():return
    rows=[]
    for entry in json.loads(source.read_text())['results']:
        if entry['mode']!='NATURAL_REPRODUCTION':continue
        stats={}
        for label,name in entry['names'].items():
            metrics=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
            timeline=json.loads((ROOT/'analysis'/(name+'_axis_timeline.json')).read_text())
            stats[label]=dict(TTFT_mean_ms=metrics['TTFT_mean_ms'],TTFT_p95_ms=metrics['TTFT_p95_ms'],
                P_queue_mean_ms=timeline['summary']['P_queue_ms']['mean'],
                P_prepare_to_ready_mean_ms=timeline['summary']['P_prepare_to_ready_ms']['mean'],
                ready_to_D_prepare_mean_ms=timeline['summary']['ready_to_D_prepare_ms']['mean'])
        baseline={k:float(np.mean([stats[label][k] for label in ('clean_before','clean_after')])) for k in stats['profile']}
        rows.append(dict(axis=entry['axis'],names=entry['names'],clean_control_mean=baseline,profile=stats['profile'],
            profile_minus_control_mean_ms={k:stats['profile'][k]-baseline[k] for k in baseline},
            evidence='DIAGNOSTIC',scope='Only source instrumentation changed. Credit, inputs, arrivals, output contract and actual batch stayed fixed. '
            'One profiled start and two clean starts do not isolate GPU-only service time or exclude ordinary start variation. '
            'A failed observer trace is not an eligible method-performance result.'))
    write(ROOT/'analysis/source_observer_overhead_intervention_v1.json',dict(results=rows,
        evidence='DIAGNOSTIC',pure_GPU_service_intervention='NOT_RUN',
        scope='This source-runtime instrumentation probe is separate from the measured credit intervention and the offline service-vector swaps.'))


if __name__=='__main__':main()
