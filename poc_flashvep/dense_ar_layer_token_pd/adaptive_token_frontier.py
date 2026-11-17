"""Refine the token balance around the coarse optimum before confirmation."""
import fcntl
import hashlib
import json
from .common import ROOT, QUIET_LOCK, resources, write
from .continuous_frontier import configuration
from .discovery import job
from .hash_qualification import gate
from ..duetserve_agg_decode.configuration import offsets


def reference_partitions():
    target=ROOT/'analysis/derived_discovery32_native_hidden_partitions.json'
    if target.exists():
        return target
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not resources().processes()
        resources().restore(list(range(8)))
        import torch
        torch.set_num_threads(1)
        folder=ROOT/'raw/Native_state_N32_O16_v1'
        source=json.loads((folder/'axis_state_prefill.json').read_text())
        for rid,value in source['requests'].items():
            hidden=torch.load(folder/f'axis_hidden_prefill_{rid}.pt',weights_only=True)
            raw=hidden.contiguous().view(torch.uint8).numpy().tobytes()
            assert hashlib.sha256(raw).hexdigest()==value['final_hidden_SHA256']
            value['partition_hashes']={}
            for suffix in (2304,2400,2560):
                offset=(8000-suffix)*4096*2
                value['partition_hashes'][str(suffix)]=dict(
                    prefix_SHA256=hashlib.sha256(raw[:offset]).hexdigest(),
                    suffix_SHA256=hashlib.sha256(raw[offset:]).hexdigest())
        source.update(evidence='TRACE_ANALYSIS',scope='Derive exact contiguous hidden partitions from preserved Native whole-prompt tensors. Check each loaded tensor against the original full hidden SHA256. No new GPU result or synthetic ready state.')
        write(target,source)
        return target


def main():
    values=[]
    try:
        reference=reference_partitions()
        for suffix in (2304,2400,2560):
            for count in (8,32):
                cfg=configuration(f'fine_Token{suffix}_standard_state_N{count}_v1',
                    count=count,cap=16,budget=128000,suffix=suffix,continuous=False,
                    correctness=True,axis_state_save_tensors=False,
                    warmup_count=0,warmup_batch_sizes=[],D_admission_batch=8,
                    ar_cudagraph_mode='NONE',axis_post_runner_witness=False)
                assert job(cfg)
                gate(cfg['name'],reference_state=reference)
        for suffix,destination in ((2304,8),(2400,4),(2400,8),(2560,8)):
            cfg=configuration(f'fine_Token{suffix}_standard_D{destination}_N128_O16_q6.4_v1',
                count=128,cap=16,budget=128000,suffix=suffix,continuous=False,
                D_admission_batch=destination,arrival_rate=6.4,
                arrival_offsets=offsets(128,6.4,2026111710),warmup_count=5,
                evaluation_arm='Token')
            value=job(cfg)
            if value:
                values.append(value)
                write(ROOT/'analysis/adaptive_token_fine_frontier.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
