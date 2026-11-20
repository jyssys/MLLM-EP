"""Confirm all four high-load cuts. Do not select only favorable pilots."""
import json
import time
from .common import ROOT,write
from .campaign import cell


def main(wait_for_pilot=True):
    path=ROOT/'configs/frozen_full200_allcuts_confirmation_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),QPS=2.35,cuts=[2,4,8,12],seeds=[2026112071,2026112072,2026112073],
            N=200,source_credit=4,hidden_source_slots=4,KV_block=128,actual_batch=[1,1,1],
            cut12='Reuse the new user-prioritized cut12 confirmation. Do not pool the historical discovery.',
            budget='Execute after the pilot matrix. Never start a new arm in the final hour. Keep unrun cells explicit.',
            scope='Every cut is confirmed at the common high load, not only the pilot winner. All failed or unfavorable starts remain visible.'))
    while wait_for_pilot and not (ROOT/'state/pilot50_all_cuts_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    rows=[];budget=json.loads((ROOT/'state/budget.json').read_text())
    for index,seed in enumerate((2026112071,2026112072,2026112073)):
        for cut in (2,4,8):
            order=('Layer','Token') if (index+cut//2)%2==0 else ('Token','Layer')
            for axis in order:
                name=f'Full200Confirm_{axis}{cut}_q2.35_c4_s{seed}_v1'
                if time.time()>=budget['last_large_dispatch_epoch']-600:
                    rows.append(dict(name=name,axis=axis,cut=cut,seed=seed,status='NOT_RUN',reason='Preserve closeout and current-run drain budget.'))
                else:
                    row=cell(name,axis,cut,seed=seed,phase='FULL200_ALLCUTS_INDEPENDENT_CONFIRMATION')
                    rows.append(row);assert row['status']!='FAILED',name
                write(ROOT/'analysis/full200_allcuts_confirmation_v1.json',dict(results=rows,evidence='MEASURED_ONLINE',
                    scope='Natural EOS full200. Same arrival starts as new Native/cut12 references. No N50 point is pooled into a full200 CI.'))
    write(ROOT/'state/full200_allcuts_confirmation_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/full200_allcuts_confirmation_hold.json',dict(time=time.time(),error=repr(error)));raise
