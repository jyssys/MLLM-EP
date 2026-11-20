"""Change one current-runtime admission or transfer setting at a time."""
import json
import time
from .common import ROOT,write
from .historical_bridge import performance,SEEDS


def main():
    path=ROOT/'configs/frozen_historical_one_factor_controls_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),N=72,input=8000,output=16,cut=8,QPS=6,BI0=True,
            seed=SEEDS[0],hidden_source_slots=32,actual_batch=[1,1,1],
            arms=[dict(KV_block=16,credit=32),dict(KV_block=128,credit=4)],
            reference='Same-seed current BI0 historical-workload arms with block128,credit32.',
            scope='One start per factor/axis. No independent-start CI. Block and credit change in separate controls. '
            'Historical A remains reused. This does not isolate every historical/current source change.'))
    while True:
        state=ROOT/'analysis/historical_workload_state_gate_BI0.json'
        main=ROOT/'analysis/historical_runtime_bridge_v1.json'
        if state.exists() and main.exists():
            gate=json.loads(state.read_text())
            if not gate['passed']:
                write(ROOT/'state/historical_one_factor_controls_complete.json',dict(status='NOT_RUN',reason='Fresh BI0 state gate failed.'));return
            rows=json.loads(main.read_text())['results']
            if all(any(r.get('axis')==axis and not r.get('BI') and r.get('status')=='MEASURED_ONLINE' for r in rows) for axis in ('Layer','Token')):break
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    results=[]
    for block,credit in ((16,32),(128,4)):
        gate=ROOT/'state/KV16_controlled_qualification_v1.json'
        for axis in ('Layer','Token'):
            if block==16 and (not gate.exists() or not json.loads(gate.read_text())['passed']):
                row=dict(axis=axis,status='NOT_RUN',reason='No valid KV16 runtime gate.')
            else:row=performance(axis,False,SEEDS[0],block=block,credit=credit)
            results.append(dict(row,KV_block=block,credit=credit))
            write(ROOT/'analysis/historical_one_factor_controls_v1.json',dict(results=results,evidence='MEASURED_ONLINE',
                scope='Same original72 prompts, same arrival trace, same Last8 and Output16. One current runtime factor changes per control. No N1 winner CI.'))
            assert row['status']!='FAILED',row
    write(ROOT/'state/historical_one_factor_controls_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/historical_one_factor_controls_hold.json',dict(time=time.time(),error=repr(error)));raise
