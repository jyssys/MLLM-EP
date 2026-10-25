"""CPU fixtures test accounting only. They are not research-model measurements."""
from collections import Counter
from types import SimpleNamespace
import torch

from poc_flashvep.looped_moe_headroom.adapters import quality
from poc_flashvep.looped_moe_headroom.probes import Controller
from poc_flashvep.looped_moe_headroom.study import compare_frames

def fixture(mode='oracle_drop',q=.5,**kwargs):
    a=SimpleNamespace(recurrent={'site'},device='cpu')
    c=Controller(a,mode,q,**kwargs)
    ids=torch.tensor([[0,1],[1,0],[0,1]])
    c.router_state['site']=dict(ids=ids,weights=torch.full((3,2),.5),x=torch.ones(3,2),pass_index=1)
    c.oracle={('site',1):dict(ids=ids,contribution=torch.arange(6).view(3,2).float())}
    x=torch.ones(6,2)
    return c,x

def test_actual_rows_are_omitted_before_arithmetic():
    c,x=fixture()
    seen=[]
    def original(inp,counts):
        seen.append((len(inp),sum(counts)))
        return inp*2
    out=c.execute('site',x,[3,3],original)
    assert seen==[(3,3)]
    assert c.work['omitted_edges']==3
    assert c.work['executed_edges']==3
    assert (out.sum(-1)==0).sum()==3

def test_q0_calls_full_native_arithmetic():
    c,x=fixture(q=0.)
    seen=[]
    def original(inp,counts):
        seen.append(len(inp))
        return inp*2
    assert torch.equal(c.execute('site',x,[3,3],original),x*2)
    assert seen==[6]

def test_random_control_uses_treatment_budget():
    c,x=fixture(mode='random_drop',matched_budget={('site',1):2})
    c.execute('site',x,[3,3],lambda inp,counts:inp*2)
    assert c.work['omitted_edges']==2
    assert c.work['executed_edges']==4

def test_cache_reset_removes_previous_request():
    c,_=fixture()
    c.previous['site']={'sentinel':'old request'}
    c.calls['site']=3
    c.reset(128)
    assert not c.previous and not c.calls and not c.router_state
    assert c.sample[-1]==127

def test_reuse_matches_token_and_expert_not_router_slot():
    c,x=fixture(mode='reuse',q=1.)
    current=c.router_state['site']['ids']
    old_ids=current.flip(-1)
    old_outputs=torch.arange(12,dtype=torch.float32).reshape(3,2,2)
    c.previous['site']=dict(ids=old_ids,outputs=old_outputs,x=torch.ones(3,2))
    calls=[]
    out=c.execute('site',x,[3,3],lambda inp,counts:calls.append(len(inp)))
    order=current.flatten().argsort(stable=True)
    expected=old_outputs.flip(1).reshape(6,2)[order]
    assert torch.equal(out,expected)
    assert calls==[]
    assert c.work['omitted_edges']==6

def test_delta_guard_uses_current_and_previous_input():
    c,x=fixture(mode='reuse',q=1.,guard=0.)
    current=c.router_state['site']['ids']
    c.previous['site']=dict(ids=current,outputs=torch.ones(3,2,2),x=torch.full((3,2),2.))
    calls=[]
    def original(inp,counts):
        calls.append(len(inp))
        return inp*3
    out=c.execute('site',x,[3,3],original)
    assert calls==[6] and c.work['omitted_edges']==0
    assert torch.equal(out,x*3)

def test_explicit_shifted_ce():
    logits=torch.tensor([[[4.,0.,0.],[0.,4.,0.],[0.,0.,4.]]])
    ids=torch.tensor([[2,0,1]])
    result=quality(logits,logits,ids)
    expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,3),ids[:,1:].reshape(-1))
    assert abs(result['nll']-expected)<1e-6
    assert result['token_denominator']==2 and result['delta_nll']==0

def test_spatial_comparison_excludes_nonadjacent_positions():
    frame=dict(ids=torch.tensor([[0,1],[0,1],[0,1]]),x=torch.ones(3,2),
        outputs=torch.ones(3,2,2),weights=torch.full((3,2),.5),positions=torch.tensor([0,1,9]))
    result=compare_frames(frame,frame,'spatial')
    assert result['positions']==1
    assert result['matched_edges']==2

def test_no_parameter_tensor_write_in_interventions():
    import inspect
    from poc_flashvep.looped_moe_headroom import probes
    source=inspect.getsource(probes)
    assert '.weight.data' not in source and '.copy_(' not in source

def test_known_model_scope_only():
    from poc_flashvep.looped_moe_headroom.common import MODELS
    assert set(MODELS)=={'ifm_ouro','ifm_huginn','foil1'}
    assert MODELS['foil1']=='ShourenWSR/how-to-loop-moe'

def test_independent_site_random_control_preserves_budget():
    c,x=fixture(mode='random_drop',matched_budget={('site',1):2},random_scope='independent_site')
    c.execute('site',x,[3,3],lambda inp,counts:inp*2)
    assert c.work['omitted_edges']==2 and c.work['executed_edges']==4

def test_primary_arms_do_not_change_after_control_followup():
    from poc_flashvep.looped_moe_headroom.study import arms
    assert len(arms())==27
    assert len(arms(include_independent=True))==31

def test_agreement_uses_integer_denominator():
    logits=torch.zeros(1,230,3)
    ids=torch.zeros(1,230,dtype=torch.long)
    result=quality(logits,logits,ids)
    assert result['top1_agreement']==1
    assert result['top1_equal_count']==result['logit_position_denominator']==230

if __name__=='__main__':
    import json
    tests={k:v for k,v in globals().copy().items() if k.startswith('test_') and callable(v)}
    for name,fn in tests.items():
        fn()
        print('PASS',name)
    print(json.dumps(dict(passed=len(tests),failed=0,scope='CPU accounting fixtures, not pretrained quality evidence')))
