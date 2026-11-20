"""Test P activity observation under the qualified fast control runtime."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .online_profile import summarize_triplet


def main():
    before='Confirmation_ControlledNative0_N200_q2.35_s2026112051_BI0_v2'
    receipt=ROOT/'raw'/before/'dispatcher_result.json'
    design=ROOT/'configs/frozen_fast_control_P_profile_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),clean_before=before,N=200,QPS=2.35,arrival_seed=2026112051,
            KV_block_tokens=128,restore_official_UCX_init=True,async_scheduling=True,
            actual_batch=[1,1,1],source_credit=4,profile_roles=['prefill'],observer_limit=.05,
            scope='Reuse the predeclared clean confirmation start. Add one profile and one clean start. '
            'Measure P activity only. D quiescence remains a host/current-stream witness, not all-stream physical idle.'))
    while not receipt.exists():
        for marker in ('STOP_REQUESTED.json','confirmation_hold.json','controlled_discovery_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    assert json.loads(receipt.read_text())['complete']
    names=dict(clean_before=before)
    for label in ('profile','clean_after'):
        name=f'FastControl_POnly_N200_q2.35_{label}_BI0_v1';names[label]=name
        serving(name,'Native','longbench_shortanswer200_v2.json',200,2.35,2026112051,
            controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',
            async_scheduling=True,KV_block_tokens=128,restore_official_UCX_init=True,
            quiescent_witness=True,kernel_activity=label=='profile',kernel_activity_roles=['prefill'],
            kernel_activity_seconds=90,phase='FAST_CONTROL_P_ONLY_OBSERVER_INTERVENTION')
        analyze(name)
    result=summarize_triplet(2.35,names,roles=('prefill',))
    result['measured_roles']=['prefill']
    result['scope']='One profiled and two clean independent starts. Frozen 5% point observer gate. P activity only. No active-decode insertion or safe future-window guarantee.'
    write(ROOT/'analysis/BI0_fast_control_P_only_observer_v1.json',result)
    write(ROOT/'state/fast_control_P_profile_complete.json',dict(time=time.time(),complete=True,observer_passed=result['observer_gate_passed']))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/fast_control_P_profile_hold.json',dict(time=time.time(),error=repr(error)));raise
