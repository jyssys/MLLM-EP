"""Download only the pinned, authorized dense-AR control."""
import importlib.metadata
from huggingface_hub import HfApi,snapshot_download
from .common import *

def main():
    setup();repo,revision=QWEN_PIN;info=HfApi().model_info(repo,revision=revision)
    assert info.sha==revision
    folder=Path(snapshot_download(repo_id=repo,revision=revision,max_workers=4,allow_patterns=[
        'config.json','generation_config.json','tokenizer*','vocab.json','merges.txt',
        'special_tokens_map.json','model*.safetensors','model*.safetensors.index.json']))
    cfg=json.loads((folder/'config.json').read_text())
    assert cfg['model_type']=='qwen3' and cfg['torch_dtype']=='bfloat16'
    write(ROOT/'configs/dense_ar_checkpoint.json',dict(repo=repo,revision=revision,path=str(folder),config=cfg,
        config_SHA256=digest(folder/'config.json'),serialized_parameter_count=info.safetensors.total,
        parameter_count_scope='HF serialized tensor count. Tied embedding aliases can change unique loaded parameter count.',
        files={p.name:p.stat().st_size for p in folder.iterdir() if p.is_file()},
        vllm_version=importlib.metadata.version('vllm'),time=time.time()))
    print('PINNED_DENSE_AR_CHECKPOINT_READY',folder,flush=True)

if __name__=='__main__':main()
