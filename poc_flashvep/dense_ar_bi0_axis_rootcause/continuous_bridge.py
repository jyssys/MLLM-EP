"""Audit the continuous bridge before claiming a continuous axis result."""
import json
import time
from .common import ROOT,write
from .campaign import cell


def main():
    declaration=ROOT/'configs/frozen_continuous_bridge_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),qualification_N=8,qualification_QPS=8,cut=8,
            contract='Correctness-only continuous request admission. No C0 whole-request D barrier. Natural EOS cap2048.',
            source_credit='Not enforced by the stock continuous path. Record this difference explicitly.',
            rule='A failing mixed-prefill/decode or lifetime test blocks C3 axis performance. Do not remove the assertion to obtain a timing.',
            baseline='C3 Native and Agg use their normal continuous scheduler. Same resource/B/S is not actual-batch matching.',
            scope='This is a bridge qualification, not evidence of decode-safe dynamic insertion.'))
    while not (ROOT/'state/credit_sweep_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    rows=[]
    for axis in ('Layer','Token'):
        name=f'C3Qualification_{axis}8_N8_q8_BI0_v1'
        row=cell(name,axis,8,dataset='qualification8.json',count=8,rate=8,seed=2026112088,
            controlled_pipeline=False,correctness_only=True,continuous_qualification_only=True,
            qualified_controlled_performance=False,axis_state_capture=True,hidden_source_slots=16,
            phase='CONTINUOUS_AXIS_CORRECTNESS_ONLY')
        rows.append(row)
        write(ROOT/'analysis/C3_axis_bridge_qualification_v1.json',dict(results=rows,evidence='CORRECTNESS_ONLY',
            scope='A successful process alone does not qualify all C3 mixed shapes. Full C3 KV, hidden and numerical gates remain required before performance.'))
    for axis in ('Native','Agg'):
        for family in ('MATCHED_NO_COMPILE','OFFICIAL_DEFAULT'):
            name=f'C3Bridge_{axis}_{family}_N200_q2.35_s2026112071_v1'
            row=cell(name,axis,rate=2.35,seed=2026112071,controlled_pipeline=False,
                execution_family=family,phase='CONTINUOUS_BRIDGE_BASELINE')
            rows.append(row)
            write(ROOT/'analysis/C3_continuous_bridge_results_v1.json',dict(results=rows,evidence='MEASURED_ONLINE',
                scope='One start per baseline configuration. C0 remains separately reported. Actual geometry and source-admission contracts differ. '
                'Failed C3 axis qualification never becomes a performance curve.'))
            assert row['status']!='FAILED',name
    write(ROOT/'state/continuous_bridge_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/continuous_bridge_hold.json',dict(time=time.time(),error=repr(error)));raise
