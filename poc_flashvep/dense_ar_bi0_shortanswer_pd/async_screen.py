"""Separate a slow mechanism-control execution path from offload effects."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze


def main():
    declaration=ROOT/'configs/frozen_C0_async_intervention_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=32,QPS=2.35,seed=2026112033,KV_block=128,
            B=8192,S=1024,actual_batch=[1,1,1],source_credit=4,restore_official_UCX_init=True,
            intervention='Only async_scheduling changes False to True. Compile0/full-decode graph stays fixed.',
            hypothesis='Synchronous engine submission contributes to the11ms controlled Decode path.',
            falsifier='Async-on does not reduce Decode intervals. A remaining gap may be compiled-kernel or other host work.',
            next_gate='Any faster async path needs exact same-geometry KV/hidden/output and lifetime qualification before Layer/Token performance.',
            scope='Diagnostic intervention. No Layer/Token performance or main winner selection.'))
    results=[]
    for enabled in (False,True):
        name=f'AsyncAB_Native_C0_A{int(enabled)}_N32_q2.35_BI0_v1'
        measured=serving(name,'Native','qualification32.json',32,2.35,2026112033,
            controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',
            async_scheduling=enabled,KV_block_tokens=128,restore_official_UCX_init=True,
            phase='CONTROLLED_ASYNC_INTERVENTION')
        analyze(name);results.append(dict(async_scheduling=enabled,name=name,metrics=measured))
        write(ROOT/'analysis/BI0_C0_async_intervention_v1.json',dict(results=results,
            evidence='MEASURED_DIAGNOSTIC',scope='One independent start per setting. Same trace, geometry and KV layout. Not a final5% noninferiority result.'))
    false,true=[r['metrics'] for r in results]
    proposed=true['mean_TBT_ms']<.95*false['mean_TBT_ms'] and true['TPOT_p95_ms']<.95*false['TPOT_p95_ms']
    write(ROOT/'state/async_screen_complete.json',dict(time=time.time(),complete=True,
        faster_candidate=bool(proposed),mean_TBT_ratio=true['mean_TBT_ms']/false['mean_TBT_ms'],
        TPOT_p95_ratio=true['TPOT_p95_ms']/false['TPOT_p95_ms'],
        scope='A speed candidate is not correctness qualification or an adopted shared runtime.'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/async_screen_hold.json',dict(time=time.time(),error=repr(error)));raise
