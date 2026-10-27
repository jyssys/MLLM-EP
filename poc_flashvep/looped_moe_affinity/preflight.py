import json
import subprocess
import time
from .common import *
from . import resources

def command(args):
    r=subprocess.run(args,text=True,capture_output=True)
    return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

def main():
    setup()
    target=ROOT/'environment_and_topology.json'
    assert not target.exists(),'Do not replace the initial state'
    result=dict(time=time.time(),spec_sha256=digest(SPEC),
        head=command(['git','rev-parse','HEAD']),branch=command(['git','branch','--show-current']),
        index_tree=command(['git','write-tree']),
        gpu_list=command(['nvidia-smi','-L']),topology=command(['nvidia-smi','topo','-m']),
        inventory=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap,pci.bus_id','--format=csv']),
        processes=resources.processes(),owned_burners=resources.burners(),
        source_revisions={path:command(['git','-C',path,'rev-parse','HEAD']) for path in
            [str(XLLM),'/home/esjung/external/how-to-loop-moe-poc','/home/esjung/external/flash-attention-affinity-poc']},
        models=json.loads((ROOT.parent/'looped_moe_headroom/model_pins.json').read_text()))
    write(target,result)
    print('PREFLIGHT_SAVED',len(result['owned_burners']))

if __name__=='__main__':main()
