import json
import subprocess
import time
from .common import *

def main():
    setup()
    target=ROOT/'preflight.json'
    assert not target.exists(),'Preserve the original preflight'
    commands={'gpu_list':['nvidia-smi','-L'],
        'inventory':['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap,pci.bus_id','--format=csv'],
        'topology':['nvidia-smi','topo','-m'],'processes':['nvidia-smi','pmon','-c','1'],
        'git_head':['git','rev-parse','HEAD'],'branch':['git','branch','--show-current'],
        'preserved_index':['git','write-tree']}
    result={key:subprocess.run(cmd,capture_output=True,text=True).stdout for key,cmd in commands.items()}
    from .resources import burners,processes
    result.update(time=time.time(),owned_burners=burners(),active_cuda=processes(),
        spec_sha256=digest(SPEC),models=json.loads((PRIOR/'model_pins.json').read_text()),
        previous_result_commit='532cd9023ce900272a5403464d54e9da45a9fe72')
    write(target,result)
    print('PREFLIGHT',target)

if __name__=='__main__':main()
