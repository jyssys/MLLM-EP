"""Freeze budget priorities before policy timing and heldout measurements."""
import hashlib
import json
import time
from .common import ROOT,write,digest

def freeze():
    target=ROOT/'configs/c3_budget_priority_amendment_v4.json';assert not target.exists()
    assert not list((ROOT/'raw').glob('V2_C3PolicyTune*'))
    assert not list((ROOT/'raw').glob('V2_C3_Main*'))
    assert not list((ROOT/'raw').glob('V2_C3_Heldout*'))
    rows=json.loads((ROOT/'data/balanced400_v2_discovery.json').read_text());chosen=[]
    for bucket in range(4):
        subset=sorted((r for r in rows if r['input_bucket']==bucket),
            key=lambda r:hashlib.sha256(('v2-policy-calibration96:'+r['prompt_id']).encode()).hexdigest())[:24]
        assert len(subset)==24;chosen+=subset
    chosen.sort(key=lambda r:hashlib.sha256(('v2-policy-order96:'+r['prompt_id']).encode()).hexdigest())
    dataset='c3_policy_calibration96_v4.json';write(ROOT/'data'/dataset,chosen)
    write(target,dict(time=time.time(),policy_dataset=dataset,policy_dataset_SHA256=digest(ROOT/'data'/dataset),
        policy_requests=96,policy_input_counts=[24]*4,selection_uses_performance_or_EOS=False,
        main400_dataset_unchanged=True,geometry_discovery208_unchanged=True,
        primary_confirmation_QPS=2.0,heldout_point_lanes={'2.0':1,'4.0':2,'4.8':3},primary_replay_lane=2,
        natural_state_observer=dict(N32='GPU SHA256 leaf tree with CPU verification over every observed byte',
            N400='Same full-byte GPU SHA256 leaf tree',strict_N8='Unchanged raw CPU SHA256'),
        execution_order=['Main400 N1 at all8 frozen QPS points','Main400 knee additional2 starts and heldout192 knee3 starts on separate pairs',
            'Original overload confirmations on all available pairs','Triggered primary-knee controlled replay before optional overload replay'],
        reason='Fresh compiled C3 actors require substantial initialization. Preserve the full Main400 and all frozen QPS. '
            'Use the same deterministic discovery-only96 calibration requests for every policy at all3 tuning loads. '
            'Keep every N8/N32/N400 correctness gate. GPU SHA256 is a validated full-byte observer,not a weaker sample hash. '
            'Prioritize independent knee confirmation before repeated overload points. Retain every planned point and report any budget-limited NOT_RUN.',
        scope='This changes calibration size and execution order before policy timing or heldout performance. '
            'It does not change client metrics,SLO,Main400,model,precision or any candidate execution limit. '
            'The96-request calibration is not the Main workload. Do not pool different pairs or request counts as independent starts.'))
    print('BUDGET_PRIORITY_FROZEN',digest(target),flush=True)

if __name__=='__main__':freeze()
