"""Small accounting fixtures. These are not model proxies or research evidence."""
import torch
from .runtime import owner_ids

def test_ragged_ownership():
    for world in [4,8]:
        ids=[e for rank in range(world) for e in owner_ids(25,world,rank)]
        assert sorted(ids)==list(range(25)) and len(ids)==len(set(ids))
        sizes=[len(owner_ids(25,world,r)) for r in range(world)]
        assert max(sizes)-min(sizes)==1

def test_top2_duplicate_rows_and_scores():
    ids=torch.tensor([[1,1],[2,0],[3,1]])
    weights=torch.tensor([[.25,.75],[.4,.6],[.2,.8]])
    x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    expected=(x[:,None,:]*(ids[:,:,None]+1)*weights[:,:,None]).sum(1)
    restored=torch.zeros_like(x)
    for rank in range(4):
        local_ids=ids[rank::4]
        order=((local_ids.flatten()%4)*4+local_ids.flatten()).argsort(stable=True)
        send=x[rank::4][order//2]
        result=send*(local_ids.flatten()[order,None]+1)
        unperm=result[order.argsort(stable=True)].reshape(-1,2,2)
        restored[rank::4]=(unperm*weights[rank::4,None].transpose(1,2)).sum(1)
    assert torch.allclose(restored,expected)

def test_no_cross_site_or_future_pass_merge():
    def legal(a,b):
        return a['site']==b['site'] and a['expert']==b['expert'] and a['request']!=b['request'] and a['ready'] and b['ready']
    a=dict(site=2,expert=4,request=0,ready=True,loop=1)
    assert legal(a,dict(a,request=1,loop=3))
    assert not legal(a,dict(a,loop=3))
    assert not legal(a,dict(a,request=1,ready=False))
    assert not legal(a,dict(a,request=1,site=3))

def main():
    tests=[test_ragged_ownership,test_top2_duplicate_rows_and_scores,test_no_cross_site_or_future_pass_merge,
        test_replica_assignment_preserves_exact_work]
    for test in tests:
        test()
        print('PASS',test.__name__)

def test_replica_assignment_preserves_exact_work():
    from .oracles import waterfill
    import numpy as np
    resident=[[0,1],[1],[2,3],[3]]
    load=waterfill([99,7,65,1],resident,4)
    assert load.sum()==172
    assert np.all(load>=0)

if __name__=='__main__':main()
