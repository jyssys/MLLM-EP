"""Close GPU work and verify actual generation on all approved devices."""
import fcntl,json,re,subprocess,time
from .common import *


def main():
    preserve();owner=resources();log=ROOT/'burner.log';start=log.stat().st_size if log.exists() else 0
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        active=owner.processes();owned=owner.burners()
        assert all(os.getsid(r['pid'])==owned.get(r['gpu']) for r in active),'An experiment or unknown GPU process is still active'
        write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),start_epoch=START,
            GPU_experiments_closed=True,elapsed_wall_hours=(time.time()-START)/3600,
            scope='No new GPU measurements may start after this receipt. Report, archive verification, and publication follow.'))
        owner.restore(list(range(8)))
    deadline=time.monotonic()+240;rounds={};samples=[]
    while time.monotonic()<deadline:
        owned=owner.burners();active=owner.processes()
        assert all(os.getsid(r['pid'])==owned.get(r['gpu']) for r in active),'Unknown GPU owner remains untouched'
        if log.exists():
            with log.open('rb') as f:f.seek(start);tail=f.read().decode(errors='replace')
            for gpu,count,tokens in re.findall(r'\[GPU (\d+)\] rounds=(\d+) generated_tokens=(\d+)',tail):
                rounds[int(gpu)]=dict(rounds=int(count),generated_tokens=int(tokens))
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        values=[dict(GPU=int(a),utilization_percent=int(b),memory_MiB=int(c)) for a,b,c in (x.split(',') for x in query.splitlines())]
        samples.append(dict(time=time.time(),GPUs=values))
        actual={r['gpu'] for r in active}
        if actual==set(range(8)) and set(rounds)==set(range(8)) and all(v['generated_tokens']>0 for v in rounds.values()):break
        missing=[g for g in range(8) if g not in actual and g not in owned]
        if missing:owner.restore(missing)
        time.sleep(5)
    owned=owner.burners();active=owner.processes();passed={r['gpu'] for r in active}==set(range(8)) and set(rounds)==set(range(8))
    snapshot=ROOT/'state/burner_closeout_snapshot.log'
    if log.exists():
        with log.open('rb') as f:f.seek(start);snapshot.write_bytes(f.read())
    result=dict(time=time.time(),passed=passed,owned_worker_sessions=owned,actual_GPU_processes=active,
        fresh_completed_generation=rounds,NVML_samples=samples,preservation=preservation(),
        user_checkout_and_index_preserved=True,external_runtime_preserved=True,
        scope='Each GPU has an approved worker and an actual GPU process in that worker session. Fresh completed generation lines confirm work, not only process existence. Missing workers alone are retried.')
    write(ROOT/'state/resource_closeout.json',result);preserve();assert passed,'Some GPU burners have not completed fresh generation'
    print('ALL_EIGHT_BURNERS_GENERATING',json.dumps(rounds),flush=True)


if __name__=='__main__':main()
