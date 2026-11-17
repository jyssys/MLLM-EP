"""Tune static fractions and intermediate token portions before cap64 freeze."""
import fcntl
import hashlib
import json
from .common import ROOT,QUIET_LOCK,resources,write
from .axis_configuration import base
from .continuous_frontier import configuration
from .hash_qualification import gate
from .discovery import job


def partitions():
    target=ROOT/'analysis/derived_discovery32_token1280_1536_partitions.json'
    if target.exists():
        return target
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not resources().processes();resources().restore(list(range(8)))
        import torch
        torch.set_num_threads(1)
        folder=ROOT/'raw/Native_state_N32_O16_v1'
        source=json.loads((folder/'axis_state_prefill.json').read_text())
        for rid,value in source['requests'].items():
            hidden=torch.load(folder/f'axis_hidden_prefill_{rid}.pt',weights_only=True)
            raw=hidden.contiguous().view(torch.uint8).numpy().tobytes()
            assert hashlib.sha256(raw).hexdigest()==value['final_hidden_SHA256']
            value['partition_hashes']={}
            for suffix in (1280,1536):
                at=(8000-suffix)*4096*2
                value['partition_hashes'][str(suffix)]=dict(
                    prefix_SHA256=hashlib.sha256(raw[:at]).hexdigest(),
                    suffix_SHA256=hashlib.sha256(raw[at:]).hexdigest())
        source.update(evidence='TRACE_ANALYSIS',scope='Exact byte partitions of preserved original Native tensors. No new GPU ready state or latency.')
        write(target,source)
    return target


def main():
    assert not (ROOT/'state/candidate_confirmation_freeze_O64.json').exists()
    values=[]
    try:
        for layers,fraction in ((8,.5),(12,.25),(12,.5)):
            cfg=base(f'O64_fraction_Last{layers}_f{fraction:g}_P8_D8_N64_q4.8_v2',
                method='Static',count=64,cap=64,rate=4.8,layers=layers,
                migration_fraction=fraction,matched=True,P_group_size=8,P_slots=4,
                D_admission_batch=8,warmup_batch_sizes=[1,2,3,4],
                hidden_source_slots=32,hidden_destination_slots=32,
                phase='DISCOVERY_CAP64_STATIC_FRACTION_REFINEMENT')
            value=job(cfg)
            if value:
                values.append(value);write(ROOT/'analysis/output64_fraction_token_refinement.json',values)
        reference=partitions()
        for suffix in (1280,1536):
            for count in (8,32):
                cfg=configuration(f'O64_refine_Token{suffix}_state_N{count}_v1',
                    count=count,cap=64,budget=128000,suffix=suffix,continuous=False,
                    correctness=True,axis_state_save_tensors=False,warmup_count=0,
                    warmup_batch_sizes=[],D_admission_batch=8,ar_cudagraph_mode='NONE',
                    axis_post_runner_witness=False)
                assert job(cfg);gate(cfg['name'],reference_state=reference)
            cfg=base(f'O64_refine_Token{suffix}_P4_D8_N64_q4.8_v1',method='Token',
                count=64,cap=64,rate=4.8,suffix=suffix,D_admission_batch=8,
                D_wait_for_source_macro=False,D_combine_ready_macros=True,
                phase='DISCOVERY_CAP64_TOKEN_PORTION_REFINEMENT')
            value=job(cfg)
            if value:
                values.append(value);write(ROOT/'analysis/output64_fraction_token_refinement.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
