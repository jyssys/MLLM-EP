"""Read-only handoff check. Never stop another user's jobs or start a burner."""
import datetime
import subprocess

from poc_dispatch_combine_asymmetry.contract import ROOT, MASK, write


def query(*args):
    return subprocess.check_output(['nvidia-smi',*args],text=True)


def main():
    gpu=query('--id='+MASK,'--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader')
    uuids={r.split(',')[1].strip() for r in gpu.splitlines()}
    apps=query('--query-compute-apps=gpu_uuid,pid','--format=csv,noheader')
    allowed=[r for r in apps.splitlines() if r.split(',')[0].strip() in uuids]
    write(ROOT/'artifacts/resource_handoff.json',{'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'allowed_physical_gpus':[0,1,4,5],'gpu_snapshot':gpu,'allowed_compute_processes':allowed,
        'burner_policy':'STOPPED_AND_DO_NOT_RESTART: latest user requested handoff after measurements',
        'resource_mutation_in_this_check':False,'other_gpu_processes':'untouched',
        'GPU_MEASUREMENT_REMAINING':False,'snapshot_status':'EMPTY' if not allowed else 'OTHER_OR_REMAINING_WORKLOAD_PRESENT'})
    if allowed:raise RuntimeError('GPUs no longer empty; read-only check did not terminate any process')
    print('HANDOFF_VERIFIED_EMPTY',MASK)


if __name__=='__main__':main()
