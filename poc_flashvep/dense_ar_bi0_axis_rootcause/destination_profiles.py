"""Measure eager destination suffixes without claiming decode-graph kernels."""
import json
import time
from .common import ROOT,write
from .campaign import cell
from .profile_analysis import triplet


def main():
    declaration=ROOT/'configs/frozen_D_suffix_profile_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),cut=4,N=200,QPS=2.35,seed=2026112061,
            output='Natural EOS cap2048',source_credit=4,hidden_source_slots=4,actual_batch=[1,1,1],
            observer_limit=.05,attribution_minimum=.99,controls=2,
            scope='D suffix kernels use API correlation for prompt-query engine steps. Decode graph enclosures are separate. '
            'Any graph inside a suffix step blocks exclusive suffix attribution. Full query geometry and outputs must match. '
            'The7-field observer gate includes P service and D prepare-to-first output. No D decode-exclusive claim.'))
    while not (ROOT/'state/natural_Last4_reproduction_complete.json').exists() or not (ROOT/'state/runtime_api_correlation_witness_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    results=[]
    for axis in ('Token','Layer'):
        prior=json.loads((ROOT/'analysis'/f'natural_Last4_{axis}_profile_triplet.json').read_text())
        names=dict(clean_before=prior['names']['clean_after'])
        for label in ('profile','clean_after'):
            name=f'DSuffixProfile_{axis}4_{label}_q2.35_s2026112061_N200_v1';names[label]=name
            row=cell(name,axis,4,seed=2026112061,
                entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
                worker_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_worker',
                kernel_activity=label=='profile',correlated_profile=label=='profile',runtime_api_correlation=True,
                kernel_activity_roles=['decode'],kernel_activity_seconds=7200,activity_copies=True,
                phase='D_EAGER_SUFFIX_KERNEL_ATTRIBUTION')
            assert row['status']!='FAILED',name
        results.append(dict(axis=axis,**triplet(names,'decode')))
        write(ROOT/'analysis/D_suffix_profile_qualification_summary.json',dict(results=results))
    write(ROOT/'state/D_suffix_profiles_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/D_suffix_profiles_hold.json',dict(time=time.time(),error=repr(error)));raise
