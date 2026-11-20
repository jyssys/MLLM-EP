"""Reduce observer exposure, not the acceptance thresholds."""
import json
import time
from .common import ROOT,write
from .campaign import cell
from .profile_analysis import triplet


def main():
    declaration=ROOT/'configs/frozen_sparse_profile_fallback_v1.json'
    sample=[f'q{i}' for i in range(64,114)]
    if not declaration.exists():
        write(declaration,dict(time=time.time(),sample_request_ids=sample,N_profiled=50,N_served=200,
            cut=4,QPS=2.35,seed=2026112061,output='Natural EOS cap2048',observer_limit=.05,
            fallback='Run only if full Runtime+Driver attribution and shape/output checks pass but its observer gate fails.',
            change='Collect the fixed contiguous request range64..113. Keep the complete200-request serving trace unchanged. No sample is selected by latency or output.',
            scope='One fallback triplet per failed axis. No repeated search for a passing run. Qualified kernel means describe these50 requests, not all200. '
            'Compare Layer and Token only on the same sampled request IDs. Observer gates still use all200 requests.'))
    while not (ROOT/'state/runtime_api_profiles_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    declarations=json.loads((ROOT/'analysis/runtime_api_profile_triplets.json').read_text())['results'];results=[]
    for axis in ('Layer','Token'):
        prior=next(r for r in declarations if r['axis']==axis and r['mode']=='NATURAL_REPRODUCTION')
        gate=triplet(prior['names'])
        if gate['qualified'] or not gate['attribution_gate'] or not gate['matching_shapes_and_strict_outputs']:
            results.append(dict(axis=axis,status='NOT_RUN',reason='Fallback is not needed, or its attribution prerequisites failed.',full_profile_gate=gate))
        else:
            names=dict(clean_before=prior['names']['clean_after'])
            for label in ('profile','clean_after'):
                name=f'SparsePProfile_{axis}4_{label}_q2.35_s2026112061_N200_v1';names[label]=name
                row=cell(name,axis,4,seed=2026112061,
                    entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
                    worker_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_worker',
                    kernel_activity=label=='profile',correlated_profile=label=='profile',runtime_api_correlation=True,
                    profile_sample_request_ids=sample,kernel_activity_roles=['prefill'],kernel_activity_seconds=7200,
                    activity_copies=True,phase='FROZEN_SOURCE_PROFILE_SUBSAMPLE')
                assert row['status']!='FAILED'
            results.append(dict(axis=axis,status='MEASURED_ONLINE',names=names,gate=triplet(names)))
        write(ROOT/'analysis/sparse_P_profile_fallback_v1.json',dict(results=results))
    write(ROOT/'state/sparse_P_profiles_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/sparse_P_profiles_hold.json',dict(time=time.time(),error=repr(error)));raise
