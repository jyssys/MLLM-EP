"""Record public environment metadata without dumping private process environments."""
import datetime
import subprocess
from poc_depth_cdb_factorial.contract import *

def cmd(args,cwd=PROJECT):
    return subprocess.check_output(args,cwd=cwd,text=True)

def main():
    write(ROOT/'artifacts/environment.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        GPU_inventory=cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv']),
        topology=cmd(['nvidia-smi','topo','-m']),
        processes=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv']),
        driver=cmd(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']),
        CDB_SHA=cmd(['git','rev-parse','HEAD'],CDB).strip(),CDB_clean=cmd(['git','status','--porcelain'],CDB)=='',
        project_base=cmd(['git','rev-parse','HEAD']).strip(),
        dependency_reuse='The pinned v2 runtime configures isolated Transformers 5.9.0 and FlashAttention3 dependencies.',
        original_spec_sha256=sha(SPEC),model_config=json.loads((MODEL/'config.json').read_text()),
        physical_to_logical={str(g):0 for g in GPUS},TP=1,EP=1,PP=1,DP=4,
        foreign_process_policy='Observe only. Never signal another user or a process on physical GPUs 0,1,2,3.'))

if __name__=='__main__':main()
