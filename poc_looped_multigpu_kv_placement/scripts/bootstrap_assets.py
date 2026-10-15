"""Download pinned public assets outside Git. Preserve all existing environments."""
import argparse
import datetime
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from poc_looped_multigpu_kv_placement.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',action='store_true');args=parser.parse_args()
    from huggingface_hub import snapshot_download
    target=ROOT/'results/env_and_revisions.json'
    if not target.exists():
        shutil.copyfile(SPEC,ROOT/'authoritative_spec.md')
        original=Path('/home/esjung/MLLM-EP-github')
        import importlib.metadata as metadata
        write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            spec_sha256=sha(SPEC),parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            mask=MASK,GPUS=GPUS,UUIDS=UUIDS,
            original_diff_sha256={label:hashlib.sha256(subprocess.check_output(cmd,cwd=original)).hexdigest()
                for label,cmd in [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]},
            nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
            topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
            installed_versions={p:metadata.version(p) for p in ['vllm','torch','transformers','huggingface-hub']},
            VLLM_PYTHON=VLLM_PYTHON,REFERENCE_PYTHON=REFERENCE_PYTHON,models=MODELS))
    def download(item):
        name,spec=item
        info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+spec['repo']+'/revision/'+spec['revision'],timeout=30))
        assert info['sha']==spec['revision']
        write(ROOT/f'results/{name}_HF_metadata.json',info)
        patterns=['*.json','*.py','*.txt','*.md']+(['*.safetensors'] if args.weights else [])
        path=snapshot_download(spec['repo'],revision=spec['revision'],allow_patterns=patterns,max_workers=4)
        print(name,path,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,MODELS.items()))


if __name__=='__main__':main()
