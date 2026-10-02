import numpy as np
import pytest
from poc_speculative_moe_ep.analysis import accept_prefix,route_features,critical


def test_greedy_off_by_one():
    assert accept_prefix([10,11,12],[10,11,99])['accepted']==2
    assert accept_prefix([10,11],[10,11,12])['bonus']
    assert accept_prefix([10],[99])['correction']
    with pytest.raises(ValueError):accept_prefix([10],[10])


def test_dedup_and_rejected_attribution():
    # Four experts/rank destinations, two same-destination experts: dedup.
    ids=np.array([[0,1,32,33,64,65,96,97]]*3)
    empty=np.empty((0,8),dtype=np.int64)
    f=route_features([ids,empty,empty,empty],[3,0,0,0],accepted=1)
    assert f['valid_paths']==24 and f['logical_dispatch_bytes']==3*4*4096
    assert f['remote_routed_paths']==18
    assert f['rejected_logical_dispatch_bytes']==4*4096


def test_idle_dp_not_replication():
    ids=np.array([[0,1,2,3,4,5,6,7]])
    f=route_features([ids,ids,ids,ids],[1,0,0,0],accepted=0)
    assert f['token_rows']==1 and f['dummy_padding_paths']==24


def test_one_critical_rank():
    frames=[]
    for rank in range(4):
        d,e,c=[(10,1,1),(1,20,1),(1,1,3),(1,1,1)][rank]
        frames.append(dict(target_forward_ms=100,layers=[dict(dispatch_ms=d,expert_ms=e,combine_ms=c,moe_ms=d+e+c)]*48))
    got=critical(frames)
    assert got['critical_rank']==1 and got['dispatch_ms']==48
    assert got['expert_ms']==960 and got['combine_ms']==48
