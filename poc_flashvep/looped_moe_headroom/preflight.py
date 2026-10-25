"""Read-only inventory and provenance. Do not treat free memory as model feasibility."""
import json
import os
import platform
import subprocess
from .common import ROOT, SPEC, XLLM, digest, write
from .resources import burners, processes

def command(words):
    p=subprocess.run(words,text=True,capture_output=True)
    return dict(command=words,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)

def main():
    inventory=dict(gpu_query=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap,pstate,temperature.gpu,driver_version','--format=csv']),
        topology=command(['nvidia-smi','topo','-m']),gpu_list=command(['nvidia-smi','-L']),
        processes=processes(),owned_burners=burners(),
        single_device_only=True,authorized_physical_gpus=list(range(8)))
    write(ROOT/'hardware_inventory.json',inventory)
    write(ROOT/'environment_preflight.json',dict(host=platform.platform(),node=platform.node(),uid=os.getuid(),
        spec_sha256=digest(SPEC), cuda_compiler=command(['/usr/local/cuda-12.8/bin/nvcc','--version']),
        repository=command(['git','rev-parse','HEAD']),branch=command(['git','branch','--show-current']),
        preserved_index_tree=command(['git','write-tree']),
        xllm=command(['git','-C',str(XLLM),'rev-parse','HEAD']),
        foil=command(['git','-C','/home/esjung/external/how-to-loop-moe-poc','rev-parse','HEAD']),
        source_status=command(['git','-C',str(XLLM),'status','--short'])))
    print('PREFLIGHT_RECORDED')

if __name__=='__main__':
    main()
