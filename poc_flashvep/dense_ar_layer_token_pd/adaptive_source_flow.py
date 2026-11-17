"""Do not impose fixed full-batch waiting on the main dynamic candidate."""
import fcntl
from .common import ROOT,QUIET_LOCK,resources,write
from .axis_configuration import base
from .discovery import job
from .layer_qualification import gate


def config(name,count=128,horizon=2,**changes):
    return base(name,method='Dynamic',count=count,rate=6.4,
        flow_balance_policy=True,flow_adaptive_source=True,flow_horizon_groups=horizon,
        flow_cost_model='empirical_host_cost_model_discovery_v2.json',
        preplanned_source_groups=False,hidden_source_slots=32,hidden_destination_slots=32,
        **changes)


def main():
    values=[]
    try:
        cfg=config('adaptive_source_flow_state_N32_v1',count=32,
            correctness_only=True,clean_timing=False,verify_layer_execution=True,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
            hidden_verify_hash=True,verify_native_prefix_KV=True,negotiation_timeout_s=120.)
        assert job(cfg);gate(cfg['name'])
        for horizon in (2,3):
            cfg=config(f'adaptive_source_flow_H{horizon}_N128_O16_q6.4_v1',horizon=horizon)
            value=job(cfg)
            if value:
                values.append(value);write(ROOT/'analysis/adaptive_source_flow_discovery.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
