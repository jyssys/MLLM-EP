"""Do not hash startup shapes. Preserve full measured-request state coverage."""
import copy
from .common import *


def main():
    assert RESUMPTION is not None
    initialize();status=json.loads((ROOT/'state/active.json').read_text());active=status.get('name')
    assert active
    jobs=[]
    for model in ('ouro','huginn'):
        for family,count in (('mixed',8),('mixed',32),('token',8),('token',32)):
            if family=='mixed' and count==8:source=f'r1_{model}_pinned_dynamic_state8'
            else:source=f'r1_{model}_actual_{family}_state{count}'
            cfg=copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
            cfg.update(name=f'r3_{model}_actual_{family}_state{count}',state_pinned_slots=4,
                state_probe_measured_only=True,phase='MEASURED_WINDOW_FULL_BYTE_STATE_QUALIFICATION_NOT_PERFORMANCE',
                observer_window_ablation=dict(reference=source,only_runtime_change='state_probe_measured_only',
                    scope='Execute the same model and graph warm-up. Hash every measured request, but not startup requests.'))
            save(cfg);jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    deferred=[name for name in queue[index:] if name.startswith('r2_')]
    write(ROOT/'state/observer_window_deferrals.json',dict(time=time.time(),configurations=deferred,
        reason='Avoid hashing dozens of startup shapes before a measured request. Preserve the earlier configurations and active run.'))
    write(path,queue[:index]+jobs+[name for name in queue[index:] if name not in deferred])
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
