"""Replay original weights through the explicit Serving projection API."""
import hashlib
import inspect
import json
import sys
from . import isolated
from .common import ROOT, write, digest


def transformed_source():
    source = inspect.getsource(isolated.main)
    before = 'operation=lambda x=x,weight=weight:torch.nn.functional.linear(x,weight)'
    after = 'operation=lambda x=x,weight=weight:linear_batch_invariant(x,weight)'
    assert source.count(before) == 1
    return source.replace(before, after)


def main():
    from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    cfg = json.load(open(sys.argv[1]))
    source = transformed_source()
    assert 'return linear_batch_invariant(x, layer.weight, bias)' in inspect.getsource(UnquantizedLinearMethod.apply)
    write(ROOT/'raw'/cfg['name']/'projection_API_audit.json', dict(
        actual_isolated_projection_API=linear_batch_invariant.__module__+'.'+linear_batch_invariant.__qualname__,
        serving_dispatch_source=inspect.getsource(UnquantizedLinearMethod.apply),
        isolated_executed_source_SHA256=hashlib.sha256(source.encode()).hexdigest(),
        installed_batch_invariant_SHA256=digest(inspect.getsourcefile(linear_batch_invariant)),
        installed_linear_SHA256=digest(inspect.getsourcefile(UnquantizedLinearMethod)),
        original_isolated_API='torch.nn.functional.linear',
        evidence='STRUCTURAL_ACCOUNTING',
        scope='The explicit unquantized Serving path under VLLM_BATCH_INVARIANT uses linear_batch_invariant. H100 enable_batch_invariant_mode does not replace aten::linear. This new namespace changes only the isolated operation call. It does not change any Serving forward or historical evidence.'))
    scope = dict(isolated.__dict__, linear_batch_invariant=linear_batch_invariant)
    exec(compile(source, __file__+':actual_projection_API', 'exec'), scope)
    scope['main']()


if __name__ == '__main__':
    main()
