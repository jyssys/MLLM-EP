"""Release-safe queue for the reference executor and actual P2P transfers."""
import subprocess
import sys
import time
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    deadline=time.monotonic()+18000
    while True:
        marker=ROOT/'results/physics_progress.json'
        if marker.exists():
            progress=read(marker)
            if any(r['exit_code'] for r in progress):raise RuntimeError('Preserve physical failure; do not launch another GPU job')
            complete=ROOT/'results/physics_completion.json'
            if complete.exists() and read(complete).get('success'):break
        if time.monotonic()>deadline:raise TimeoutError('Physical queue did not finish within the bounded wait')
        time.sleep(20)
    jobs=[('reference_components_v1','4',[CPU,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.check_causal_kv_dependencies']),
        ('stage_transfers_v1','4,5,6,7',[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            '--module','poc_looped_exact_depth_chunk_pipeline.scripts.measure_stage_transfers'])]
    for name,mask,cmd in jobs:
        command=[sys.executable,'-m','poc_looped_exact_depth_chunk_pipeline.scripts.gpu_session',
            '--gpus',mask,'--timeout','1200','--output','results/'+name,'--']+cmd
        subprocess.run(command,cwd=PROJECT,check=True)


if __name__=='__main__':main()
