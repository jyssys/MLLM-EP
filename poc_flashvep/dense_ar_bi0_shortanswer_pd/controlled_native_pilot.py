"""Qualify a separate batch1 Native mechanism control, not a new baseline default."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .online_profile import summarize_triplet


def main():
    declaration=ROOT/'configs/frozen_controlled_Native_pilot_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=32,QPS=2.35,seed=2026112033,
            actual_batches=[1,1,1],source_credit=4,B=8192,S=1024,
            execution='Compile0, full decode graph, async=False',
            scope='Separate mechanism control. The official continuous Native and Agg baselines remain unchanged. '
            'D admission requires an idle certificate. P computation does not wait for a destination reservation.'))
    names={}
    for label in ('clean_before','profile','clean_after'):
        name=f'ControlledNative_N32_q2.35_{label}_BI0_v1';names[label]=name
        result=serving(name,'Native','qualification32.json',32,2.35,2026112033,
            controlled_pipeline=True,source_credit_limit=4,
            execution_family='MATCHED_NO_COMPILE',async_scheduling=False,
            phase='CONTROLLED_NATIVE_OBSERVER_CONTROL',quiescent_witness=True,
            kernel_activity=label=='profile',activity_copies=True,kernel_activity_seconds=40)
        analyze(name)
        folder=ROOT/'raw'/name
        certs=[json.loads(line) for line in (folder/'idle_certificates.jsonl').read_text().splitlines()]
        assert certs and all(r['passed'] for r in certs)
        for role in ('prefill','decode'):
            rows=json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']
            assert rows and all(r['requests']==1 for r in rows)
        assert result['quality_gate']['point_passed'],'Keep a failed quality screen separate from final qualification'
    result=summarize_triplet(2.35,names)
    write(ROOT/'analysis/ControlledNative_observer_gate_v1.json',result)
    write(ROOT/'state/controlled_Native_pilot_complete.json',dict(time=time.time(),complete=True,
        observer_passed=result['observer_gate_passed']))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/controlled_Native_pilot_hold.json',dict(time=time.time(),error=repr(error)));raise
