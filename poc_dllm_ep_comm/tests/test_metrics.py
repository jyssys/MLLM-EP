import numpy as np
from poc_dllm_ep_comm.metrics import traffic,bootstrap_mean,decision,graph_stats

def test_destination_dedup_not_topk_bytes():
    ids=np.array([[0,1,2,3,64,65,128,192],[64,65,66,67,68,69,70,71]])
    t=traffic(ids,0)
    assert t['remote_paths'].tolist()==[4,8]
    assert t['dispatch'].tolist()==[3*4096,4096]
    assert t['total'].sum()==8*4096

def test_local_paths_have_zero_remote_bytes():
    t=traffic([[64,65,66,67,68,69,70,71]],1)
    assert t['total'].sum()==0 and t['local_paths'].sum()==8

def test_bootstrap_unit_is_request_not_layer():
    stat=bootstrap_mean([2,2,2])
    assert stat['n']==3 and stat['ci95']==[2,2]

def test_gate_does_not_promote_unvalidated_timing():
    assert decision(4,.3,False)=='CONDITIONAL'
    assert decision(4,.05,True)=='NO-GO'
    assert decision(1.2,.4,True)=='NO-GO'
    assert decision(4,.3,True)=='GO'
    assert decision(4,.3,True,False)=='RUNTIME_BLOCKED'

def test_dense_graph_and_rank_imbalance():
    s=graph_stats(np.ones((4,4)),[8,8,8,8])
    assert s['edge_density']==1 and s['rank_max_mean']==1 and s['rank_cv']==0
