"""Record package versions and small dependency hashes without a CUDA context."""
import importlib.metadata
import platform
import subprocess
import sys
from .common import ROOT,MODEL,FORK,UPSTREAM,write,digest

def main():
    versions={}
    for name in ('torch','transformers','numpy','triton','flash-attn-4','vllm'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='NOT_INSTALLED'
    files=[p for p in MODEL.iterdir() if p.suffix in ('.json','.safetensors')]
    write(ROOT/'dependency_lock.json',dict(python=sys.version,platform=platform.platform(),
        executable=sys.executable,packages=versions,runtime_pin=UPSTREAM,
        runtime_diff=subprocess.check_output(['git','diff','--binary'],cwd=FORK,text=True),
        model_revision=MODEL.name,
        model_files=[dict(name=p.name,bytes=p.stat().st_size,resolved_blob_name=p.resolve().name,
            sha256=digest(p) if p.suffix=='.json' else 'Hub blob identity recorded. No second full-weight scan.') for p in files],
        counters_tool='/opt/nvidia/nsight-compute/2025.1.1/ncu',
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=index,driver_version','--format=csv,noheader'],text=True)))

if __name__=='__main__':main()
