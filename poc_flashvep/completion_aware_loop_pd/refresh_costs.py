"""Repeat qualified serial continuation costs without changing the primary workload."""
import copy
from .prepare import prepare
from .common import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        width=4096 if model=='ouro' else 2048
        names=[f'p_{model}_tile{width}_loops1_serial8']
        names += [f'pf_{model}_loops{n}_quantum1_serial8' for n in ((2,3) if model=='ouro' else (2,4,8))]
        names += [f'p2_{model}_tile{q}_loops2_quantum1_serial8' for q in (128,256)]
        names += [f'pw_{model}_tile{q}_loops2_quantum1_serial8' for q in (512,1024)]
        for old in names:
            source=OLD/'configs'/(old+'.json')
            assert source.exists(),old
            cfg=copy.deepcopy(json.loads(source.read_text()))
            cfg.update(name='cost_refresh_events_'+old,phase='REFRESHED_SERIAL_CUDA_EVENT_COST_NOT_ONLINE_HEADROOM',
                reused_qualified_configuration=str(source),calibration_reference=old,
                continue_after_failure=True,native_transport_limits=dict(shutdown_timeout=120),
                kernel_activity_seconds=0.,sparse_device_trace=True,sparse_decode_period=31,full_kernel_trace=False)
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/refreshed_cost_jobs.json',jobs);print(' '.join(jobs))


if __name__=='__main__':main()
