"""Test one observer change: avoid waiting for four CPU snapshot slots."""
import copy
from .common import *


def main():
    assert RESUMPTION is not None
    initialize();jobs=[]
    status=json.loads((ROOT/'state/active.json').read_text());active=status.get('name')
    assert active,'Insert the diagnostic only between registered queue entries'
    deferred={f'r1_{model}_actual_token_state{count}' for model in ('ouro','huginn') for count in (8,32)}
    for model in ('ouro','huginn'):
        for family,count in (('mixed',32),('token',8),('token',32)):
            old=f'r1_{model}_actual_{family}_state{count}'
            cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
            cfg.update(name=f'r2_{model}_actual_{family}_state{count}',state_pinned_slots=count+2,
                phase='FULL_BYTE_OBSERVER_CAPACITY_DIAGNOSTIC_NOT_CLEAN_PERFORMANCE',
                observer_capacity_ablation=dict(reference=old,only_runtime_change='state_pinned_slots',
                    previous_slots=4,new_slots=count+2,
                    prediction='GPU progress does not wait for the four-slot CPU hash queue. Exact state and native lifetime remain unchanged.'))
            # The host has more than 1.7 TiB available at preparation time.
            # This diagnostic reserves less than 0.4 TiB for the largest pair.
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            assert available>600*2**30,'Do not allocate large pinned observer buffers under host memory pressure'
            save(cfg);jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    pending=[name for name in queue[index:] if name not in deferred]
    assert all(not (ROOT/'raw'/name).exists() for name in deferred),'Do not replace a started diagnostic'
    write(ROOT/'state/observer_capacity_deferrals.json',dict(time=time.time(),configurations=sorted(deferred),
        reason='Use cohort-sized diagnostic buffers. Preserve all original immutable configurations.'))
    write(path,queue[:index]+jobs+pending)
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
