import torch
from poc_dllm_attn_moe_pipeline.attention import partial_state,merge_states


def test_exact_global_softmax_not_sum_of_outputs():
    torch.manual_seed(7)
    q,k,v = torch.randn(1,2,3,8),torch.randn(1,2,9,8),torch.randn(1,2,9,8)
    mask = torch.rand(1,1,3,9) > .3
    a,b = partial_state(q,k[:,:,:5],v[:,:,:5],mask[...,:5]),partial_state(q,k[:,:,5:],v[:,:,5:],mask[...,5:])
    merged = merge_states(a,b)
    reference = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    torch.testing.assert_close(merged,reference,atol=1e-6,rtol=1e-5)


def test_empty_and_all_masked():
    q,k,v = torch.randn(1,1,2,4),torch.randn(1,1,3,4),torch.randn(1,1,3,4)
    empty = partial_state(q,k[:,:,:0],v[:,:,:0])
    blocked = partial_state(q,k,v,torch.zeros(1,1,2,3,dtype=torch.bool))
    assert torch.equal(merge_states(empty,blocked),torch.zeros_like(q))


def test_stale_current_kv_changes_result():
    q,k,v = torch.ones(1,1,2,4),torch.randn(1,1,4,4),torch.randn(1,1,4,4)
    prefix = partial_state(q,k[:,:,:2],v[:,:,:2])
    correct = merge_states(prefix,partial_state(q,k[:,:,2:],v[:,:,2:]))
    stale = merge_states(prefix,partial_state(q,k[:,:,2:]+8,v[:,:,2:]-8))
    assert not torch.allclose(correct,stale)
