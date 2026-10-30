"""CPU contract tests. GPU numerical regressions use the real-operand probes."""
import torch
from types import SimpleNamespace
from .norm_probe import fixed_norm


def test_fixed_norm_preserves_shape_dtype_and_formula():
    layer=SimpleNamespace(weight=torch.ones(5280,dtype=torch.bfloat16),variance_epsilon=1e-6)
    for size in (1,2,3,4,8,16,32):
        x=torch.randn(size,5280).to(torch.bfloat16)
        actual=fixed_norm(layer,x)
        value=x.float()
        expected=layer.weight*(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
        assert actual.shape==x.shape and actual.dtype==x.dtype
        assert torch.allclose(actual,expected,atol=.04,rtol=.01)


def test_padding_does_not_mix_rows():
    layer=SimpleNamespace(weight=torch.ones(2048,dtype=torch.bfloat16),variance_epsilon=1e-6)
    x=torch.randn(3,2048).to(torch.bfloat16)
    together=fixed_norm(layer,x)
    separately=torch.cat([fixed_norm(layer,row[None,:]) for row in x])
    assert torch.equal(together,separately)


def test_greedy_argmax_is_not_topk_tie_order():
    x=torch.tensor([1.,2.,2.,0.])
    assert x.argmax().item()==1
    assert x.topk(2).values.tolist()==[2.,2.]


def test_frozen_failure_has_equal_history_before_divergence():
    import json
    from .common import ROOT
    values=json.loads((ROOT/'configs/failure_manifest.json').read_text())
    for row in values:
        n=row['first_divergence']
        assert row['input_equal'] and row['history_equal']
        assert row['left_row']['output_ids'][:n]==row['right_row']['output_ids'][:n]
        assert row['left_row']['output_ids'][n]!=row['right_row']['output_ids'][n]


def test_queued_compute_retains_fixed_depth_event_and_byte_guards():
    from .transfer_probe import queued_compute_ready
    request=SimpleNamespace(request_id='x',sampling_params=SimpleNamespace(min_loops=4,max_loops=4))
    event=SimpleNamespace(query=lambda:True)
    transfer=SimpleNamespace(request=request,outgoing=True,cancelled=False,last_event=event,chunks=['still queued'])
    connector=SimpleNamespace(transfers={'x':transfer},cache=SimpleNamespace(max_loops=4),
                              transport=SimpleNamespace(inflight_bytes=0),config=SimpleNamespace(max_inflight_bytes=1024))
    assert queued_compute_ready(connector,request)
    event.query=lambda:False
    assert not queued_compute_ready(connector,request)
    event.query=lambda:True
    connector.transport.inflight_bytes=1024
    assert not queued_compute_ready(connector,request)
    connector.transport.inflight_bytes=0
    transfer.cancelled=True
    assert not queued_compute_ready(connector,request)
    transfer.cancelled=False
    transfer.outgoing=False
    assert queued_compute_ready(connector,request)
    transfer.outgoing=True
    request.sampling_params.min_loops=2
    import pytest
    with pytest.raises(ValueError,match='full fixed depth'):
        queued_compute_ready(connector,request)


def test_completed_resume_rejects_changed_contract(monkeypatch,tmp_path):
    from . import run
    from .common import write
    import pytest
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run,'setup',lambda:None)
    write(tmp_path/'configs/example.json',dict(name='example',gpus=[0,1]))
    write(tmp_path/'raw/example/status.json',dict(status='COMPLETE'))
    with pytest.raises(ValueError,match='contract conflict'):
        run.execute(dict(name='example',gpus=[2,3]))
