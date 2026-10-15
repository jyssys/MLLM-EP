"""Inspect authentic checkpoint tensors and pinned source without initializing weights."""
import struct
import subprocess
from poc_looped_multigpu_kv_placement.contract import *


def headers(path):
    result={}
    for part in sorted(path.glob('*.safetensors')):
        with part.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0];entries=json.loads(stream.read(size))
        result.update({key:dict(**value,file=part.name) for key,value in entries.items() if key!='__metadata__'})
    assert result,'No real pretrained checkpoint tensors'
    return result


def main():
    result={}
    for name in MODELS:
        path=model_path(name);config=json.loads((path/'config.json').read_text());tensors=headers(path)
        layer_prefix='model.layers.' if name=='ouro' else 'transformer.'
        indices=sorted({int(k.split('.')[2]) for k in tensors if k.startswith(layer_prefix) and name=='ouro'})
        result[name]=dict(model=MODELS[name],config=config,config_sha256=sha(path/'config.json'),
            actual_checkpoint_tensors=len(tensors),checkpoint_tensor_bytes=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in tensors.values()),
            checkpoint_tensors=tensors,stored_ouro_layer_indices=indices,
            remote_source_sha256={p.name:sha(p) for p in path.glob('*.py')})
    official=Path('/home/esjung/external/looped-kv-huginn-official')
    result['huginn_official_plugin']=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip(),
        attention_source_sha256=sha(official/'vllm/raven_vllm.py'),
        native_head_count=55,TP2_divisible=55%2==0,TP4_divisible=55%4==0,
        limitation='Official RavenAttention floor-divides head counts. Native QKVParallelLinear and config require exact divisibility. No padding/remapping supplied.',
        modern_API='Plugin imports vllm.attention.layer. Native0.26 uses vllm.model_executor.layers.attention. Do not invent a new executor.')
    write(ROOT/'results/architecture_cache_audit.json',result)
    print('Pinned real weight metadata and official code audited.')


if __name__=='__main__':main()
