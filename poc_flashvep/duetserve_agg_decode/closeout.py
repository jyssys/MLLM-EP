"""Close GPU work only after agent review and verified eight-device restoration."""
import fcntl
import json
import os
import subprocess
import time

from .common import QUIET_LOCK, ROOT, preserve, resources, write


def main():
    assert json.loads((ROOT / 'state/analysis_ready_for_review.json').read_text())['status'] == 'READY_FOR_AGENT_REVIEW'
    assert json.loads((ROOT / 'state/agent_review_receipt.json').read_text())['passed']
    assert not (ROOT / 'state/CAMPAIGN_CLOSED.json').exists()
    preserve()
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        for row in owner.processes():
            assert os.getsid(row['pid']) == owner.burners().get(row['gpu']), ('UNKNOWN_GPU_OWNER_UNTOUCHED', row)
        owner.restore(list(range(8)))
        samples, active_gpus = [], set()
        until = time.monotonic() + 120
        while time.monotonic() < until:
            active, approved = owner.processes(), owner.burners()
            assert all(os.getsid(r['pid']) == approved.get(r['gpu']) for r in active)
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,power.draw',
                                              '--format=csv,noheader,nounits'], text=True)
            values = []
            for line in output.splitlines():
                gpu, util, memory, power = (p.strip() for p in line.split(','))
                values.append(dict(GPU=int(gpu), utilization_percent=int(util), memory_MiB=int(memory), power_W=float(power)))
                if int(util) > 0 and int(gpu) in {r['gpu'] for r in active}:
                    active_gpus.add(int(gpu))
            samples.append(dict(time=time.time(), utilization=values, processes=active, approved_worker_sessions=approved))
            if {r['gpu'] for r in active} == set(range(8)) and active_gpus == set(range(8)):
                break
            time.sleep(1)
        assert {r['gpu'] for r in owner.processes()} == set(range(8))
        assert active_gpus == set(range(8)), 'Burner PIDs exist, but actual activity is not verified on every device'
        preserve()
        write(ROOT / 'state/GPU_resource_closeout.json', dict(time=time.time(), passed=True,
            GPU_ids=list(range(8)), observed_active_GPU_ids=sorted(active_gpus), samples=samples,
            user_checkout_index_external_runtime_preserved=True,
            scope='Approved run_utilize.sh workers remain detached on all eight GPUs. Positive sampled NVML activity verifies restoration, not experimental physical kernel utilization.'))
        live_log = ROOT / 'burner.log'
        if live_log.exists():
            with live_log.open('rb') as stream:
                size = stream.seek(0, 2)
                stream.seek(max(0, size - 65536))
                tail = stream.read().decode(errors='replace')
            write(ROOT / 'state/burner_closeout_log_snapshot.json', dict(time=time.time(), tail=tail,
                scope='Snapshot of the approved burner log. The live log can continue to grow after publication.'))
        write(ROOT / 'state/CAMPAIGN_CLOSED.json', dict(time=time.time(), status='CLOSED',
            GPU_work_complete=True, scientific_success_claim=False,
            resources_restored=True, scope='Closeout permits archival and private-index publication. It does not change any result gate.'))
    print('ALL_EIGHT_BURNERS_ACTIVE_AND_USER_STATE_PRESERVED', flush=True)


if __name__ == '__main__':
    main()
