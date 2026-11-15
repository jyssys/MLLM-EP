"""Test a load-generator storage hypothesis without changing GPU work."""
import fcntl,gc,time
from .common import *
from .input_pool import intern


def main():
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        cfg=json.loads((ROOT/'configs/sustain_discovery_Native_O16_q2p8_s51_v1.json').read_text())
        rows=json.loads((ROOT/'data'/cfg['dataset']).read_text())
        raw_objects=len({id(r['input_ids']['qwen8']) for r in rows});before=[]
        for _ in range(5):
            start=time.perf_counter();gc.collect(2);before.append(1000*(time.perf_counter()-start))
        normalized=intern(rows);assert normalized==rows
        del rows;gc.collect(2);after=[]
        for _ in range(5):
            start=time.perf_counter();gc.collect(2);after.append(1000*(time.perf_counter()-start))
        value=dict(requests=len(normalized),unchanged_prompt_tokens=True,
            distinct_raw_token_lists=raw_objects,distinct_interned_token_lists=len({id(r['input_ids']['qwen8']) for r in normalized}),
            before_full_collection_host_ms=before,after_full_collection_host_ms=after,
            evidence='MEASURED_CPU_MICROBENCHMARK',
            scope='Run between GPU jobs under the shared resource lease. This tests plausibility, not the cause of a past Serving pause. No prefix cache, computation reuse, GPU idle or Serving speedup is claimed.')
        write(ROOT/'analysis/input_pool_CPU_GC_microbenchmark.json',value)
        print('INPUT_POOL_CPU_STUDY',before,after,flush=True)
        if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
