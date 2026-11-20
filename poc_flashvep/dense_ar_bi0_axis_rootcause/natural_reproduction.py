"""Reproduce the historical Last4 trace before assigning its GPU cause."""
import time
from .common import ROOT,write
from .campaign import cell


def main():
    path=ROOT/'configs/frozen_historical_Last4_natural_reproduction_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),cut=4,QPS=2.35,seed=2026112061,N=200,
            source_credit=4,hidden_source_slots=4,KV_block=128,actual_batches=[1,1,1],
            contract='Natural EOS cap2048, same original200 request order and Poisson offsets as the historical discovery.',
            order=['clean_before','profile','clean_after'],roles=['prefill'],
            scope='Fresh reproduction under the new buffered coordinator. The historical measured2711ms queue gap is not replaced by a different-cut or fixed-output result.'))
    while not (ROOT/'state/correlated_collector_witness_complete.json').exists():time.sleep(3)
    rows=[]
    for axis in ('Layer','Token'):
        names={}
        for label in ('clean_before','profile','clean_after'):
            name=f'NaturalLast4Repro_{axis}_{label}_q2.35_c4_s2026112061_N200_v1';names[label]=name
            row=cell(name,axis,4,seed=2026112061,
                entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
                worker_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_worker',
                kernel_activity=label=='profile',correlated_profile=label=='profile',
                kernel_activity_roles=['prefill'],kernel_activity_seconds=7200,activity_copies=True,
                phase='HISTORICAL_LAST4_MATCHED_NATURAL_REPRODUCTION')
            rows.append(dict(axis=axis,label=label,result=row));assert row['status']!='FAILED',name
            write(ROOT/'analysis/natural_Last4_reproduction_v1.json',dict(results=rows,evidence='MEASURED_ONLINE',
                scope='Natural full200 reproduction and source-only profiler controls. Profile timings are not clean performance.'))
        write(ROOT/'analysis'/f'natural_Last4_{axis}_profile_triplet.json',dict(axis=axis,names=names))
    write(ROOT/'state/natural_Last4_reproduction_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/natural_Last4_reproduction_hold.json',dict(time=time.time(),error=repr(error)));raise
