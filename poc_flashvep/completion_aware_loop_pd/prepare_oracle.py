"""Prepare isolated Native full-cohort activity calibration."""
from .prepare import config,prepare
from .common import *


def main():
    prepare();jobs=[]
    assert json.loads((ROOT/'tools/deferred_activity_selftest.json').read_text())['passed']
    for model in ('ouro','huginn'):
        name=f'oracle_{model}_deferred_full128_calibration'
        cfg=config(model,name)
        cfg.update(full_kernel_trace=True,phase='ORACLE_COLLECTOR_CALIBRATION',
            native_transport_limits=dict(shutdown_timeout=300))
        save(cfg);jobs.append(name)
    write(ROOT/'state/oracle_calibration_jobs.json',jobs);print(' '.join(jobs))


if __name__=='__main__':main()
