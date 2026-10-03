import numpy as np
from poc_dllm_request_persistent_replica.scripts.analyze_persistence import jsd


def test_jsd_exact_and_finite():
    p=np.array([[.1,.2,.3,.4],[1.,0,0,0]])
    assert np.allclose(jsd(p,p),0)
    assert jsd(np.array([[1.,0]]),np.array([[0.,1.]])).item()==1
    assert np.isfinite(jsd(p,p+1e-16)).all()
