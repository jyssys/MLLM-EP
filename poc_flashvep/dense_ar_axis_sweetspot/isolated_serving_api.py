"""Replay the exact serving Linear API, rather than a generic F.linear call."""
import hashlib
import inspect
import json
import sys
from .common import ROOT,bind_legacy,write,digest


def main():
    bind_legacy()
    from ..dense_ar_matched_axis_pd import isolated
    from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
    from vllm.model_executor.layers import linear,batch_invariant
    cfg=json.loads(open(sys.argv[1]).read())
    code=inspect.getsource(isolated.main)
    before='operation=lambda x=x,weight=weight:torch.nn.functional.linear(x,weight)'
    after='operation=lambda x=x,weight=weight:linear_batch_invariant(x,weight)'
    assert code.count(before)==1
    code=code.replace(before,after)
    before='if rows in (8000,2672) else timed(operation)'
    after="if rows in cfg['isolated_profile_widths'] else timed(operation)"
    assert code.count(before)==1
    code=code.replace(before,after)
    folder=ROOT/'raw'/cfg['name']
    write(folder/'actual_operator_API_contract.json',dict(
        called_API='vllm.model_executor.layers.batch_invariant.linear_batch_invariant',
        serving_callsite='UnquantizedLinearMethod.apply in vllm.model_executor.layers.linear',
        linear_source=str(linear.__file__),linear_source_SHA256=digest(linear.__file__),
        batch_invariant_source=str(batch_invariant.__file__),batch_invariant_SHA256=digest(batch_invariant.__file__),
        original_driver_SHA256=hashlib.sha256(inspect.getsource(isolated.main).encode()).hexdigest(),
        executed_driver_SHA256=hashlib.sha256(code.encode()).hexdigest(),
        evidence='STRUCTURAL_ACCOUNTING',
        scope='Only isolated operator invocation and captured widths change. Original BF16 weights and captured activation rows stay unchanged. These measurements exclude online queue, paged attention, transport and Decode contention.'))
    scope=dict(isolated.__dict__,linear_batch_invariant=linear_batch_invariant)
    exec(compile(code,__file__+':exact_serving_API','exec'),scope)
    scope['main']()


if __name__=='__main__':
    main()
