import numpy as np
from poc_dllm_attn_moe_pipeline.scripts.estimate_schedule_headroom import bootstrap


def test_bootstrap_independent_request_count():
    result=bootstrap([10.,20.,30.])
    assert result['n_requests']==3
    assert result['mean']==20.
    assert result['ci95'][0]<=20<=result['ci95'][1]


def test_zero_exposed_window_does_not_imply_latent_timing():
    result=bootstrap(np.zeros(8))
    assert result['ci95']==[0.,0.]
    # This is an interface release invariant, not a hardware noise-floor test.
