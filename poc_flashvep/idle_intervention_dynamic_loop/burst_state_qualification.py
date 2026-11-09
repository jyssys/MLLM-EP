"""Cover real mixed decisions under a declared correctness-only arrival stress."""
import copy
from .common import *


def main():
    initialize();active=json.loads((ROOT/'state/active.json').read_text()).get('name');jobs=[]
    for model in ('ouro','huginn'):
        for count in (8,32):
            source=f'r6_{model}_actual_mixed_state{count}'
            cfg=copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
            cfg.update(name=f'r9_{model}_actual_mixed_state{count}',arrival_offsets=[0.]*count,
                arrival_rate=None,arrival_seed=None,arrival_pattern='SIMULTANEOUS_CORRECTNESS_STRESS',
                phase='REAL_ONLINE_MIXED_CUT_CORRECTNESS_STRESS_NOT_SERVING_PERFORMANCE',
                stress_scope='All request cuts still require a fresh D decision after actual GPU boundary completion. No request receives a forced cut. This arrival stress is not the primary Poisson workload.')
            cfg['native_transport_limits']=dict(cfg['native_transport_limits'],shutdown_timeout=300.)
            save(cfg)
            if count==8:jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1 if active in queue else len(queue)
    write(path,queue[:index]+jobs+queue[index:]);print(json.dumps(jobs))


if __name__=='__main__':main()
