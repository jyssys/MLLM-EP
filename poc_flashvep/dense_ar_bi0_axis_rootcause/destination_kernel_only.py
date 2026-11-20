"""Use an independent kernel-only protocol after graph-record validation fails."""
import json
import time
from .common import ROOT,write,save
from .run import execute
from .campaign import cell
from .profile_analysis import triplet


def main():
    declaration=ROOT/'configs/frozen_D_kernel_only_profile_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),cut=4,N=200,QPS=2.35,seed=2026112061,
            diagnosis='The graph-enclosure trace emitted zero-timestamp graph records. Its global integrity gate failed. Keep that trace diagnostic.',
            change='Enable RUNTIME+DRIVER+CONCURRENT_KERNEL. Do not enable GRAPH_TRACE or copy activity. Do not rebuild or replace the collector library.',
            gate='Counter integrity,>=99% eager-kernel correlation,strict full-query/output matching and all7 observer changes<=5%. '
            'The model replay witness must account for every measured graph replay and show no replay in a prompt-query step.',
            scope='Only eager suffix-step kernels are measured. Graph-child kernels are excluded from this attribution scope. '
            'No decode-graph exclusive time or DMA result is inferred. Each axis has one new triplet. No acceptance threshold changes.'))
    cfg=dict(name='RuntimeOnlyCorrelation_Witness_v1',method='Native',
        entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.correlation_witness',gpus=[1],
        batch_invariant=False,correctness_only=True,wall_timeout=180,runtime_only_correlation=True)
    assert execute(save(cfg))
    while not (ROOT/'state/D_suffix_profiles_complete.json').exists():time.sleep(5)
    prior=json.loads((ROOT/'analysis/D_suffix_profile_qualification_summary.json').read_text())['results'];results=[]
    for axis in ('Layer','Token'):
        previous=next(r for r in prior if r['axis']==axis)
        names=dict(clean_before=previous['names']['clean_after'])
        for label in ('profile','clean_after'):
            name=f'DKernelOnly_{axis}4_{label}_q2.35_s2026112061_N200_v1';names[label]=name
            row=cell(name,axis,4,seed=2026112061,
                entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
                worker_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_worker',
                kernel_activity=label=='profile',correlated_profile=label=='profile',runtime_only_correlation=True,
                kernel_activity_roles=['decode'],kernel_activity_seconds=7200,activity_copies=False,
                phase='D_EAGER_KERNEL_ONLY_OBSERVER_CONTROL')
            assert row['status']!='FAILED'
        results.append(dict(axis=axis,**triplet(names,'decode')))
        write(ROOT/'analysis/D_kernelonly_suffix_profile_qualification_summary.json',dict(results=results))
    write(ROOT/'state/D_kernelonly_profiles_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/D_kernelonly_profiles_hold.json',dict(time=time.time(),error=repr(error)));raise
