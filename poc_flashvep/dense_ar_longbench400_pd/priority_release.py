"""Pause only queue dispatch at safe boundaries,then restore declared work."""
import json
import time
from .common import ROOT,write,dispatch_allowed

QUEUES=('forward_kernel_only_case0_v1','forward_kernel_only_case1_v1',
    'guard_static_tuning_lane0_v1','guard_static_tuning_lane1_v1')


def main():
    paths=[ROOT/'state'/('PAUSE_QUEUE_'+name+'.json') for name in QUEUES]
    for path in paths:
        assert not path.exists(),'Do not replace an unrelated queue pause'
        write(path,dict(owner='ITL_COMPUTE_THROTTLE_PRIORITY_V1',time=time.time(),
            scope='Pause future dispatch only. Current GPU work drains. Preserve every log and result.'))
    print('LOWER_PRIORITY_DISPATCH_PAUSED',flush=True)
    while True:
        checks=[ROOT/'state'/f'D_compute_throttle_{v}_qualification_v1.json' for v in ('Layer','Token','Hybrid','StockToken')]
        if all(p.exists() and json.loads(p.read_text()).get('passed') for p in checks):break
        assert dispatch_allowed();time.sleep(5)
    for path in paths:
        assert json.loads(path.read_text())['owner']=='ITL_COMPUTE_THROTTLE_PRIORITY_V1'
        path.unlink()
    write(ROOT/'state/ITL_priority_release_v1.json',dict(time=time.time(),released=list(QUEUES),
        scope='Removed only owned temporary dispatch-pause markers. All experimental evidence remains intact.'))
    print('LOWER_PRIORITY_QUEUES_RESUMED',flush=True)


if __name__=='__main__':main()
