"""Inspect a pinned alternate release without downloading weights or executing its code."""
import urllib.request
from poc_looped_multigpu_kv_placement.contract import *


def fetch(url):
    with urllib.request.urlopen(url,timeout=30) as response:return response.read()


def main():
    revision='df0ed1ea415b55f0a944a291f90da9d47c98b5d5'
    model='IFM/LoopedLM-P1-dense-huginn-336b';checkpoint='e87780591bf62f453299f5056bb7f51884d84b84'
    base=f'https://raw.githubusercontent.com/ifm-ai/xllm-loop/{revision}/'
    source={name:fetch(base+name) for name in ['README.md','xllm/paper_part1/native_inference.py',
        'xllm/models/looped.py','release/paper-part1/LICENSE.weights']}
    root=f'https://huggingface.co/{model}/resolve/{checkpoint}/'
    config=json.loads(fetch(root+'config.json'));index=json.loads(fetch(root+'model.safetensors.index.json'))
    metadata=json.loads(fetch(f'https://huggingface.co/api/models/{model}/revision/{checkpoint}'))
    assert metadata['sha']==checkpoint
    inference=source['xllm/paper_part1/native_inference.py'].decode()
    write(ROOT/'results/fallback_audit.json',dict(repository='https://github.com/ifm-ai/xllm-loop',repository_revision=revision,
        checkpoint=model,checkpoint_revision=checkpoint,license=metadata.get('cardData',{}).get('license'),
        config=config,stored_tensor_bytes=index['metadata']['total_size'],
        source_sha256={name:hashlib.sha256(value).hexdigest() for name,value in source.items()},
        scope='Metadata and source audit only. No checkpoint load, output, GPU timing, or TP validation.',
        feasibility=dict(Q_heads=config['model']['num_heads'],KV_heads=config['model']['num_kv_heads'],
            ordinary_Q_TP4_divisible=config['model']['num_heads']%4==0,
            ordinary_KV_TP4_divisible=config['model']['num_kv_heads']%4==0,
            native_loader_API_present='def load_native_model(' in inference,
            runtime='Native xLLM checkpoint and model implementation, not the installed Ouro/Huginn serving executors.',
            decision='NOT_ADOPTED. A separate executor/cache integration and KV partition are outside the bounded one-family characterization.')))
    print('Pinned alternate checkpoint, license, geometry, and native-format boundary. No fake TP result.')


if __name__=='__main__':main()
