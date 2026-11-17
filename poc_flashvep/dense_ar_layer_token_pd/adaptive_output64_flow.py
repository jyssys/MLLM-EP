"""Test the calibrated variable source geometry before output64 selection."""
import fcntl
from .common import ROOT,QUIET_LOCK,resources,write
from .adaptive_source_flow import config
from .discovery import job
from .layer_qualification import gate


def main():
    values=[]
    try:
        cfg=config('adaptive_source_flow_state_N32_O64_v1',count=32,cap=64,
            request_caps=[64]*32,correctness_only=True,clean_timing=False,
            verify_layer_execution=True,full_prompt_state_witness=True,
            full_prompt_state_witness_roles=['prefill','decode'],hidden_verify_hash=True,
            verify_native_prefix_KV=True,negotiation_timeout_s=120.)
        assert job(cfg);gate(cfg['name'])
        for horizon in (2,3):
            cfg=config(f'adaptive_source_flow_H{horizon}_N64_O64_q4.8_v1',count=64,
                horizon=horizon,cap=64,request_caps=[64]*64,arrival_rate=4.8)
            from ..duetserve_agg_decode.configuration import offsets
            cfg['arrival_offsets']=offsets(64,4.8,cfg['arrival_seed'])
            value=job(cfg)
            if value:
                values.append(value);write(ROOT/'analysis/adaptive_output64_flow_discovery.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
