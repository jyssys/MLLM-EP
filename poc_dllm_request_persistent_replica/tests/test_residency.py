import numpy as np
from poc_dllm_request_persistent_replica.data import Profile
from poc_dllm_request_persistent_replica.planner import plan
from poc_dllm_request_persistent_replica.contract import EXPERT_BYTES,ROW_DC_BYTES


def block(experts,iterations=2):
    data={(t,l,s):np.empty((0,8),dtype=np.int16) for t in range(iterations) for l in range(1,20) for s in range(4)}
    for t in range(iterations):data[t,1,0]=np.array([experts],dtype=np.int16)
    return dict(iterations=iterations,data=data)


def test_copy_once_across_many_blocks():
    p=Profile([block([64,65,0,1,2,3,4,5]) for _ in range(6)])
    selected={0:{1:{64,65}},1:{},2:{},3:{}}
    r=p.evaluate(selected,1)
    assert r['copy_bytes']==2*EXPERT_BYTES
    assert r['saved_bytes']==5*2*ROW_DC_BYTES
    assert r['block_post_bytes'][0]==2*ROW_DC_BYTES
    assert r['future_blocks']==5 and r['future_iterations']==10
    assert r['replica_use'][0]['future_blocks_used']==5


def test_partial_destination_never_counts_as_saved():
    p=Profile([block([64,65,0,1,2,3,4,5]) for _ in range(3)])
    r=p.evaluate({0:{1:{64}},1:{},2:{},3:{}},1)
    assert r['saved_bytes']==0


def test_layer_identity_is_charged():
    p=Profile([block([64,65,0,1,2,3,4,5]) for _ in range(3)])
    r=p.evaluate({0:{1:{64},10:{64}},1:{},2:{},3:{}},1)
    assert r['copy_bytes']==2*EXPERT_BYTES


def test_no_future_load_guard_or_profile_leakage():
    original=Profile([block([64,65,0,1,2,3,4,5]),block([128,129,0,1,2,3,4,5])])
    changed=Profile([block([64,65,0,1,2,3,4,5]),block([192,193,0,1,2,3,4,5])])
    for cap in (.10,.15,.20,None):
        assert plan(original.subset(0,1),2,cap)==plan(changed.subset(0,1),2,cap)


def test_profile_window_load_cap():
    p=Profile([block([0,1,64,65,128,129,192,193])])
    selected=plan(p,4,.10)
    assert p.evaluate(selected)['worst_stage_load_increase']<=.10+1e-10
