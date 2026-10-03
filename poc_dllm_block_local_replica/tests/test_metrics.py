import numpy as np
from poc_dllm_block_local_replica.metrics import post_traffic,closure_groups
from poc_dllm_block_local_replica.scripts.destination_closure_oracle import bundle_greedy,lp_upper


def test_partial_destination_does_not_close():
    ids=np.array([[64,65,66,128,129,192,193,0]])
    assert post_traffic(ids,0,{64})['dc_bytes']==3*8192
    assert post_traffic(ids,0,{64,65,66})['dc_bytes']==2*8192
    assert post_traffic(ids,0,{64,65,66})['localized_paths']==3


def test_layer_identity_and_joint_memory():
    ids=np.array([[64,65,0,1,2,3,4,5]])
    groups={1:closure_groups(ids,0),10:closure_groups(ids,0)}
    chosen=bundle_greedy(groups,2)
    assert sum(map(len,chosen.values()))==2
    assert len(chosen)==1
    assert lp_upper(groups,2)==8192


def test_all_experts_still_execute():
    ids=np.array([[64,65,66,128,129,192,193,0]])
    result=post_traffic(ids,0,set(ids.flatten()))
    assert result['dc_bytes']==0
    assert result['load'].sum()==8
    assert result['load'][0]==8


def test_causal_plans_do_not_see_future():
    from poc_dllm_block_local_replica.scripts.simulate_replication import make_plan
    data={}
    for t in range(3):
        for l in range(1,20):
            for s in range(4):
                data[t,l,s]=np.array([[0,1,2,3,64,65,128,192]],dtype=np.int16)
    other={k:v.copy() for k,v in data.items()}
    for (t,l,s),v in other.items():
        if t>=2:v[:]=np.array([8,9,10,11,72,73,136,200])
    global_frequency=np.ones((19,4,256))
    for policy in ('global','first_frequency','first_destination','first2_destination'):
        original=make_plan(data,3,policy,5,global_frequency)
        changed=make_plan(other,3,policy,5,global_frequency)
        assert original==changed


def test_profiling_traffic_not_removed():
    from poc_dllm_block_local_replica.scripts.simulate_replication import evaluate_block
    data={(t,l,s):np.empty((0,8),dtype=np.int16) for t in range(2) for l in range(1,20) for s in range(4)}
    data[0,1,0]=data[1,1,0]=np.array([[64,65,0,1,2,3,4,5]])
    plans={s:{} for s in range(4)};plans[0][1]={64,65}
    result,_=evaluate_block(data,2,plans,1)
    assert result['live_bytes']==2*8192
    assert result['post_bytes']==8192


def test_load_oracle_enforces_per_stage_cap():
    from poc_dllm_block_local_replica.scripts.simulate_replication import make_plan,evaluate_block
    data={}
    for l in range(1,20):
        for s in range(4):
            data[0,l,s]=np.array([[0,1,64,65,128,129,192,193]],dtype=np.int16)
    plans,profile,_=make_plan(data,1,'oracle_load',32,np.zeros((19,4,256)))
    result,_=evaluate_block(data,1,plans,profile)
    assert result['worst_stage_load_increase']<=.10+1e-9
