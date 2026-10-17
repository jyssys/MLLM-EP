"""Scoped real communication experiments. Counter failures remain explicit."""
import os
import subprocess
from poc_recurrent_asymmetry.contract import *


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==MASK
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    commands=[('counter',["/usr/local/cuda/bin/ncu","--profile-from-start","off","--launch-count","3",
        "--metrics","dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "--csv",PYTHON,'-m','poc_recurrent_asymmetry.scripts.counter_probe'],'4'),
        ('physical',[PYTHON,'-m','poc_recurrent_asymmetry.scripts.physical_microbench'],MASK),
        ('nccl',[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',
                 'poc_recurrent_asymmetry.scripts.nccl_microbench'],MASK)]
    results=[]
    for name,command,mask in commands:
        directory=output/name;directory.mkdir(exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=mask,LOOPED_KV_OUTPUT=str(directory))
        with (directory/'console.log').open('w') as log:
            result=subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(name=name,command=command,CUDA_VISIBLE_DEVICES=mask,exit_code=result.returncode))
        write(output/'subprocess_results.json',results)
        if name!='counter':assert result.returncode==0,'Preserved physical experiment failure: '+name
    write(output/'status.json',dict(success=True,experiments=results,
        counter_status='AVAILABLE' if results[0]['exit_code']==0 else 'UNAVAILABLE. See original failure log.'))


if __name__=='__main__':main()
