"""Record installed library versions without model execution."""
import argparse
import importlib.metadata
import platform
import sys
import torch
from .contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('label');a=p.parse_args()
    libraries={}
    for name in ['torch','transformers','tokenizers','regex','vllm','numpy','safetensors','flash-attn','flash-attn-3','math-verify']:
        try:libraries[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:libraries[name]='NOT_INSTALLED_IN_BASE_PATH'
    record=dict(label=a.label,executable=sys.executable,python=platform.python_version(),libraries=libraries,
        torch_CUDA=torch.version.cuda,NCCL=torch.cuda.nccl.version(),
        CDB_runtime_override='The pinned CDB loader inserts isolated Transformers 5.9.0 and FA3 dependency paths.')
    write(ROOT/'state'/('versions_'+a.label+'.json'),record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
