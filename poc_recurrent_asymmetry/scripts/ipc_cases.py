"""Compare one allocator variable in two fresh native-runtime processes."""
import os
import json
import subprocess
from poc_recurrent_asymmetry.contract import *


def main():
    output=Path(os.environ['LOOPED_KV_OUTPUT']);results=[]
    for name,value in [('expandable','expandable_segments:True'),('cuda_malloc',None)]:
        env=dict(os.environ)
        env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        if value:env['PYTORCH_CUDA_ALLOC_CONF']=value
        command=[VLLM_PYTHON,'-m','poc_recurrent_asymmetry.scripts.ipc_probe']
        run=subprocess.run(command,cwd=PROJECT,env=env,capture_output=True,text=True)
        results.append(dict(case=name,return_code=run.returncode,stdout=run.stdout,stderr=run.stderr,
            command=command,physical_gpu_mask=env['CUDA_VISIBLE_DEVICES']))
    write(output/'cases.json',results)
    write(output/'status.json',dict(success=True,scope='Diagnostic red/green evidence, not architecture measurement.'))


if __name__=='__main__':main()
