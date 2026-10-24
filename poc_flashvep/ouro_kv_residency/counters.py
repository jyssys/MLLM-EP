"""Attempt the installed NCU tool. Preserve denied counters as missing evidence."""
import fcntl
import os
import subprocess
import time
from .common import ROOT,FORK,REPO,PYTHON,write
from .resources import stop,restore

def main():
    path=ROOT/'state'/'dispatcher.lock';path.parent.mkdir(exist_ok=True)
    lock=path.open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    output=ROOT/'counters';output.mkdir(exist_ok=True)
    stop(list(range(8)))
    results=[]
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(FORK)+':'+str(REPO),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for p in (2048,32768,65408):
            command=['/opt/nvidia/nsight-compute/2025.1.1/ncu','--target-processes','all',
                '--profile-from-start','off','--kernel-name','regex:.*flash_attn.*','--launch-count','1',
                '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed',
                '--csv','--log-file',str(output/f'P{p}.csv'),'--export',str(output/f'P{p}'),
                PYTHON,'-m','poc_flashvep.ouro_kv_residency.counter_probe','--P',str(p)]
            result=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
            write(output/f'P{p}_attempt.json',dict(command=command,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr,wall_s=time.time()))
            log=(output/f'P{p}.csv').read_text() if (output/f'P{p}.csv').exists() else ''
            results.append(dict(P=p,returncode=result.returncode,permission_denied='ERR_NVGPUCTRPERM' in log+result.stdout+result.stderr))
            write(output/'summary.json',results)
            if result.returncode or results[-1]['permission_denied']:break
    finally:restore(list(range(8)))

if __name__=='__main__':main()
