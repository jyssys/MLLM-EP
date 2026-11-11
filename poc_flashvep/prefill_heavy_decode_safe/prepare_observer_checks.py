"""Qualify each observer change separately before combining observations."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for model in ('ouro','huginn'):
        parent=ROOT/'configs'/f'native_{model}_{"early_profile" if model=="ouro" else "clean_recovery"}_state8.json'
        for mode in ('events','devicebuffers'):
            cfg=json.loads(parent.read_text());name=f'native_{model}_{mode}_observer_state8'
            cfg.update(name=name,discovery_activity=mode=='devicebuffers',activity_device_buffers=mode=='devicebuffers',
                discovery_event_observer=mode=='events',phase='SEPARATE_OBSERVER_EXACT_STATE_QUALIFICATION',
                required_qualification=f'native_{model}_{"early_profile" if model=="ouro" else "clean_recovery"}_state32',
                discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
                discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
            save(cfg);jobs.append(name)
    write(ROOT/'state/observer_qualification_plan.json',dict(jobs=jobs,
        scope='Exact Native output, valid KV, Hidden State, full work, wire identity, ACK lifetime, and normal worker exits. Not serving performance.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
