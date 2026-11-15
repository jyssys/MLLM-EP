"""Observe UCX protocol selection before changing any transfer path."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous


def main():
    name='diagnostic_UCX_protocol_default_N8_G8_Last12_v1'
    options=method_settings(wait=True)
    options.update(async_decode=True,transport_environment={'UCX_PROTO_INFO':'y','UCX_LOG_LEVEL':'info'},
        static_layers=12,migration_fraction=1.,request_caps=[2]*8,arrival_offsets=[0.]*8,
        arrival_rate=None,correctness_only=True,clean_timing=False,wall_timeout=420,
        phase='TRANSPORT_PROTOCOL_DIAGNOSTIC_NOT_CLEAN_TIMING')
    cfg=base(name,'STATIC',count=8,cap=16,rate=3.,**options)
    try:
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        lines=[]
        for role in ('prefill','decode'):
            path=ROOT/'raw'/name/(role+'.log')
            lines.extend(dict(role=role,line=line) for line in path.read_text().splitlines()
                if any(word in line for word in ('cuda_ipc','cuda_copy','rc_mlx5','UCX  INFO','UCX INFO','rendezvous','rndv')))
        write(ROOT/'analysis/UCX_default_protocol_probe.json',dict(name=name,lines=lines,
            evidence='TRACE_ANALYSIS',
            scope='Protocol logs, not a bandwidth benchmark or a serving result. NV18 topology alone does not establish which transport an operation selected. No transport setting changed in this probe.'))
        print('UCX_PROTOCOL_LINES',len(lines),flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
