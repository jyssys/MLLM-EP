"""Acquire fixed pretrained checkpoints. GPU work is separate."""
import datetime
import importlib.metadata
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from poc_looped_recurrence_tp_4models.contract import *


def main():
    from huggingface_hub import snapshot_download
    ASSETS.mkdir(parents=True,exist_ok=True)
    target=ROOT/'results/env_and_pins.json'
    assert not target.exists(),'Preserve the existing audit'
    shutil.copyfile(SPEC,ROOT/'authoritative_spec.md')
    original=Path('/home/esjung/MLLM-EP-github')
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),spec_sha256=sha(SPEC),models=MODELS,
        runtime={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','huggingface-hub']},
        interpreter=PYTHON,GPUS=GPUS,UUIDS=UUIDS,parity_contract=PARITY,
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        original_diffs={k:hashlib.sha256(subprocess.check_output(c,cwd=original)).hexdigest() for k,c in [
            ('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}))
    def download(item):
        name,model=item
        metadata=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+model['repo']+'/revision/'+model['revision'],timeout=30))
        assert metadata['sha']==model['revision']
        write(ROOT/f'results/{name}_metadata.json',metadata)
        path=snapshot_download(model['repo'],revision=model['revision'],allow_patterns=[
            '*.json','*.py','*.txt','*.md','LICENSE*','*.safetensors','*.tiktoken'],max_workers=4)
        print(name,path,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(download,MODELS.items()))


if __name__=='__main__':main()
