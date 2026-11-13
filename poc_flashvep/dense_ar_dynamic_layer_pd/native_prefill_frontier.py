"""Check larger Native P batches before claiming a strong P bottleneck."""
import fcntl
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics,compare


def main():
    initialize();rows=[]
    c=base('capacity_P_HND_lean_B4_8_16_v4',count=16,cap=16,dataset='calibration.json')
    c.update(module='capacity',capacity_mode='P_ONLY',gpus=[0],hidden_transport=None,kv_layout='HND',
        token_budget=128000,P_slots=16,D_slots=16,jobs=[dict(batch=b,repeats=2) for b in (4,8,16)],
        phase='INDEPENDENT_LARGER_NATIVE_PREFILL_CAPACITY_NOT_SERVING')
    try:
        if not execute(save(c),restore=False):return
        for cap in (16,32):
            template=json.loads((ROOT/'configs'/f'lean_native_O{cap}_N64_v1.json').read_text())
            for batch in (8,16):
                c=dict(template);c.update(name=f'frontier_Native_P{batch}_O{cap}_N64_v1',P_slots=batch,
                    token_budget=batch*8000,D_token_budget=32000,warmup_batch_sizes=[1,2,4,8] if batch==8 else [1,2,4,8,16],
                    phase='STRONG_NATIVE_LARGER_PREFILL_BATCH_VALIDATION')
                if not execute(save(c),restore=False):return
                row=dict(name=c['name'],metrics=metrics(c['name']),
                    output_gate=compare(f'coarse_O{cap}_N64_q4.777_native16_s1_v1',c['name']))
                rows.append(row);write(ROOT/'state/native_prefill_frontier_progress.json',dict(results=rows))
                print('NATIVE_PREFILL_FRONTIER',c['name'],row['output_gate']['passed'],row['metrics']['TTFT_p95_ms'],flush=True)
                if not row['output_gate']['passed']:return
        write(ROOT/'analysis/native_prefill_frontier_results.json',dict(results=rows))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
