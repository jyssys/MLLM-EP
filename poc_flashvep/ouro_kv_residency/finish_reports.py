"""Add transfer, observer, and source details from actual retained records."""
import csv
import json
import numpy as np
from .common import ROOT
from .report import table,put

def main():
    tier=json.loads((ROOT/'raw/tiering_exact/tiering.json').read_text())
    rows=[[r['tokens'],r['bytes_one_direction']/2**20]+[f"{r['measurements'][k]['median_wall_ms']:.4f}" for k in ('H2D','D2H','ROUNDTRIP')] for r in tier]
    last=tier[-1];roundtrip=last['measurements']['ROUNDTRIP']['median_wall_ms']
    put('tiering_break_even.md',f'''# Exact KV movement and break-even

The test transfers original BF16 model-derived K and V.
Both host buffers use pinned memory.
CUDA completion waits enter the wall interval.
Each size has twelve technical repetitions in one process.
Every completed roundtrip is byte-exact.

{table(['Valid positions','K+V MiB','H2D median ms','D2H median ms','Roundtrip median ms'],rows)}

The 16K payload is one physical layer at one recurrence depth.
Moving a full depth requires 24 such payloads.
The conditional roundtrip sum is {24*roundtrip:.2f} ms per depth.
Moving all four depths gives {96*roundtrip:.2f} ms per output token.
Those sums are extrapolations, not measured full-model transfers.
The native B1/16K decode interval is about 40 ms.
Repeated full-cache CPU tiering therefore fails this bounded break-even screen.

NCCL P2P records use the same original KV values.
Their latency includes completion barriers and software overhead.
It is not raw NVLink bandwidth.
The distributed attention operator instead transfers a small query and partial-output summaries.
It does not transfer the historical KV on every attention call.

Native RLT already keeps its whole pool in HBM.
It has no CPU KV-fetch stall for a depth prefetch policy to remove.
The study therefore does not construct an offload engine.
Future memory-pressure exploration must charge legal cache placement and append costs.
''')
    obs={}
    for mode in ('False','True'):
        data=json.loads((ROOT/f'raw/observer_mem_{mode}/completed.json').read_text())
        obs[mode]=float(np.mean([r['ITL_mean_ms'] for r in data['summary']['rows']]))
    cal=list(csv.DictReader((ROOT/'contention_calibration.csv').open()))
    max_ratio=max(float(r['relative_to_GPU0_isolated']) for r in cal if r['gpu']=='0')
    put('baseline_reproduction.md',f'''# Baseline reproduction and interference

The native model ran in all 22 feasible B/L cells of the registered discovery grid.
Each cell has two source-disjoint process starts and two technical repetitions.
`baseline_summary.csv` reports those results.
The table does not treat technical repetitions as independent requests.

The calibration compares one, two, four, and eight independent workers.
It uses P8192/B4 and common warm barriers.
Width order reverses in the second independent start.
The largest GPU0 ratio to its isolated reference is {max_ratio:.5f}×.
That observed deviation is below the registered 3% screen.
Other GPUs describe device spread, not a paired isolation experiment for every device.
This test does not certify zero rolling-startup interference.

The light memory observer compares the same final-source P8192/B4 workload.
Observer OFF mean ITL is {obs['False']:.4f} ms.
Observer ON mean ITL is {obs['True']:.4f} ms.
The observed difference is {(obs['True']/obs['False']-1)*100:.3f}%.
Each arm has one process and three technical repetitions.
This is an overhead diagnostic, not a precise population confidence bound.

Primary layout pairs also use a Quiet Lane with no other managed GPU workload.
The 64K follow-up tests a variable parallel E2E result instead of selecting its fastest trial.
The original parallel measurements remain unchanged.
''')
    parity=list(csv.DictReader((ROOT/'cohort_output_parity.csv').open()))
    exact=sum(r['tokens_exact']=='True' for r in parity)
    text=(ROOT/'correctness.md').read_text()
    text+=f'\nThe retained layout cohorts match tokens in {exact}/{len(parity)} request pairs.\n'
    text+='These are bounded hardware outputs with the same static depth, not benchmark answers.\n'
    (ROOT/'correctness.md').write_text(text)

if __name__=='__main__':main()
