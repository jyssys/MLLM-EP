"""Reserve a third independent held-out seed after the block16 discovery."""
import json
import time
from .common import ROOT,write
from .historical_bridge import performance


def main():
    path=ROOT/'configs/frozen_historical_block16_third_holdout_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),discovery_seed=2026111920,
            primary_confirmation_seeds=[2026111921,2026111922,2026111923],
            additional_seed=2026111923,methods=['Layer','Token','Native'],N=72,input=8000,output=16,
            QPS=6,cut=8,BI0=True,KV_block=16,credit=32,hidden_source_slots=32,actual_batch=[1,1,1],
            reason='Do not count the favorable discovery as one of three held-out confirmation starts.',
            scope='Freeze this third seed before the first block16 held-out pair completes. All outcomes remain included. No time-budget extension.'))
    while not (ROOT/'state/historical_block16_confirmation_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    gate=json.loads((ROOT/'analysis/historical_Native_block16_state_gate_v1.json').read_text())
    rows=[]
    for axis in ('Token','Native','Layer'):
        if not gate['passed']:row=dict(axis=axis,status='NOT_RUN',reason='Native state gate failed.')
        else:row=performance(axis,False,2026111923,block=16,credit=32)
        rows.append(row);write(ROOT/'analysis/historical_block16_extra_holdout_v1.json',dict(results=rows,
            evidence='MEASURED_ONLINE',scope='Third held-out arrival seed. Do not combine the seed20 discovery with the primary held-out3-start CI.'))
        assert row['status']!='FAILED',row
    write(ROOT/'state/historical_block16_extra_holdout_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/historical_block16_extra_holdout_hold.json',dict(time=time.time(),error=repr(error)));raise
