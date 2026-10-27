"""Read checkpoint tensor headers. Do not allocate a GPU or load model weights."""
from collections import Counter
from math import prod
from safetensors import safe_open
from .common import ROOT,MODEL_ASSETS,write,digest

def main():
    checkpoint=MODEL_ASSETS/'foil1/Foil-1/model.safetensors'
    counts=Counter();tensors=[]
    with safe_open(checkpoint,framework='pt',device='cpu') as source:
        for name in source.keys():
            view=source.get_slice(name);shape=view.get_shape();dtype=view.get_dtype()
            category=('rope_buffer' if name.endswith('.rope.rotations') else
                'expert' if 'experts_w' in name else 'router' if 'router.' in name else
                'attention' if '.attn.' in name else 'embedding_or_lm_head' if 'embed' in name or 'lm_head' in name else 'other')
            count=prod(shape);counts[category]+=count
            tensors.append(dict(name=name,shape=shape,dtype=dtype,elements=count,category=category,
                registered_parameter=category!='rope_buffer'))
    learned=sum(v for k,v in counts.items() if k!='rope_buffer')
    assert learned==553730048 and counts['rope_buffer']==9437184
    assert all(v['dtype']==('F32' if v['category']=='rope_buffer' else 'BF16') for v in tensors)
    write(ROOT/'quality/foil_parameter_inventory.json',dict(tensors=tensors,counts=dict(counts),learned_parameters=learned,
        persistent_tensor_elements=sum(counts.values()),checkpoint_sha256=digest(checkpoint),
        note='FP32 rope.rotations are registered buffers, not learned weights. Native module metadata confirms this classification.'))
    print(dict(counts))

if __name__=='__main__':main()
