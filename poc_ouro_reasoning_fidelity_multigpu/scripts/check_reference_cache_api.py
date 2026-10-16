"""Small CPU reproduction of the pinned cache constructor and cached length API."""
import argparse
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--version',required=True,choices=['4.54.1','4.57.1'])
args=parser.parse_args()
if args.version=='4.54.1':
    sys.path.insert(0,'/home/esjung/external/ouro-reasoning-fidelity-assets/hf454_deps')
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    import importlib
    import torch
    import transformers
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    Cache=get_class_from_dynamic_module('modeling_ouro.UniversalTransformerCache',MODEL)
    module=importlib.import_module(Cache.__module__)
    result=dict(version=transformers.__version__,source=module.__file__,
        source_sha256=sha(module.__file__),pinned_sha256=sha(MODEL/'modeling_ouro.py'))
    assert result['version']==args.version
    assert result['source_sha256']==result['pinned_sha256']
    try:
        cache=Cache(192)
        key=torch.zeros((1,16,2,128),dtype=torch.bfloat16)
        for slot in range(192):
            cache.update(key,key,slot)
        assert all(cache.get_seq_length(slot)==2 for slot in range(192))
        result.update(success=True,slots=len(cache.key_cache),mask_sizes=cache.get_mask_sizes(torch.tensor([2]),0))
    except Exception as error:
        result.update(success=False,error=repr(error))
    write(ROOT/f'results/cache_api_{args.version}.json',result)
    print(result)

if __name__=='__main__':
    main()
