"""Inspect the selected transport without changing the Main engine settings."""
import json
import os
import time
from .common import ROOT,write
from .campaign import serving


def main():
    write(ROOT/'configs/UCX_protocol_probe_design_v1.json',dict(time=time.time(),count=8,
        dataset='qualification8.json',offered_QPS=2,arrival_seed=2026112042,
        environment=dict(UCX_PROTO_INFO='y',UCX_LOG_LEVEL='info'),
        hypothesis='Distinguish reported CUDA IPC paths from host-staged or software paths. Do not infer the path from GPU topology alone.',
        scope='Log-only Native diagnostic after all Main and sensitivity controls. Not a latency comparison or a new Primary setting.'))
    while not (ROOT/'state/followup_controls_complete.json').exists():
        for file in ('STOP_REQUESTED.json','Main_driver_hold.json','followup_driver_hold.json'):
            assert not (ROOT/'state'/file).exists(),file
        time.sleep(5)
    prior={key:os.environ.get(key) for key in ('UCX_PROTO_INFO','UCX_LOG_LEVEL')}
    try:
        os.environ.update(UCX_PROTO_INFO='y',UCX_LOG_LEVEL='info')
        serving('diagnostic_Native_UCX_protocol_N8_BI1_v1','Native','qualification8.json',8,2.,2026112042,
            phase='DIAGNOSTIC_UCX_PROTOCOL_SELECTION',diagnostic_environment=dict(UCX_PROTO_INFO='y',UCX_LOG_LEVEL='info'))
    finally:
        for key,value in prior.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    write(ROOT/'state/UCX_protocol_probe_complete.json',dict(time=time.time(),complete=True,
        scope='Output gate passed. Inspect protocol logs before claiming a transport-path result. Diagnostic timing is not Main timing.'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/UCX_protocol_probe_hold.json',dict(time=time.time(),error=repr(error)))
        raise
