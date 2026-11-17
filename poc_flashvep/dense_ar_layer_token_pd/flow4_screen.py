"""Test the same visible-work policy without forcing decode batch8."""
import fcntl
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job
from .layer_qualification import gate


def main():
    values=[]
    try:
        cfg=base('flow4_GPU_state_qualification_N32_v1',method='Dynamic',count=32,
            matched=True,flow_balance_policy=True,flow_horizon_groups=2,
            flow_cost_model='empirical_host_cost_model_discovery_v2.json',
            correctness_only=True,clean_timing=False,verify_layer_execution=True,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
            hidden_verify_hash=True,verify_native_prefix_KV=True,negotiation_timeout_s=120.)
        assert job(cfg)
        gate(cfg['name'])
        for horizon in (2,3):
            cfg=base(f'flow4_H{horizon}_N128_O16_q6.4_v1',method='Dynamic',count=128,
                rate=6.4,matched=True,flow_balance_policy=True,flow_horizon_groups=horizon,
                flow_cost_model='empirical_host_cost_model_discovery_v2.json',
                hidden_source_slots=32,hidden_destination_slots=32)
            value=job(cfg)
            if value:
                values.append(value)
                write(ROOT/'analysis/flow4_discovery_screen.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
