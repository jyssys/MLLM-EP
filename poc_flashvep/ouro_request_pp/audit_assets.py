"""Read checkpoint metadata on CPUs. Do not allocate a CUDA device."""
import collections
import json
from safetensors import safe_open
from .contract import ROOT,MODEL,REVISION,sha,write


def main():
    tensors=[]
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as f:
        for name in f.keys():
            value=f.get_slice(name)
            shape=value.get_shape();dtype=value.get_dtype()
            count=1
            for d in shape:count*=d
            tensors.append(dict(name=name,shape=shape,dtype=dtype,elements=count))
    dtype_counts=dict(collections.Counter(t['dtype'] for t in tensors))
    config=json.loads((MODEL/'config.json').read_text())
    write(ROOT/'checkpoint_audit.json',dict(model='ByteDance/Ouro-1.4B',revision=REVISION,
        checkpoint_sha256=sha(MODEL/'model.safetensors'),checkpoint_bytes=(MODEL/'model.safetensors').stat().st_size,
        tensor_dtypes=dtype_counts,parameters=sum(t['elements'] for t in tensors),
        config=config,tensors=tensors,license='Apache-2.0, official model card research-purpose notice',
        original_source_sha256=sha(MODEL/'modeling_ouro.py')))
    print('ASSET_AUDIT',len(tensors),dtype_counts)


if __name__=='__main__':main()
