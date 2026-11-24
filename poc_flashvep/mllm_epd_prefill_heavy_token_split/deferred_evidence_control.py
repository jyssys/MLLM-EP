"""One-factor client evidence timing control on the existing Native EPD group."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time
import urllib.request

from .client import execute,load_replay_counts,resolve_manifest
from .common import ROOT,digest,preserve,write
from .metrics import summarize
from .source_snapshot import snapshot

URL='http://127.0.0.1:18845'
WORKERS=['metadata_B3_E4','metadata_B3_P5','metadata_B3_D6']


def readiness():
    leases=[]
    for name in WORKERS:
        path=ROOT/'state'/f'server_{name}.json'
        row=json.loads(path.read_text())
        assert row['ready'] and not row['finished']
        process=Path('/proc',str(row['pid']))
        assert process.exists() and process.stat().st_uid==os.getuid()
        with urllib.request.urlopen(f'http://127.0.0.1:{row["port"]}/health',timeout=3) as response:
            assert response.status==200
        leases.append(dict(name=name,pid=row['pid'],GPU_info=row['GPU_info'],
            source_archive_SHA256=row['source_archive_SHA256']))
    return leases


async def correctness(name):
    path=ROOT/'data/calibration_manifest_template_v2.json'
    values=json.loads(path.read_text())['requests'];rows=[values[i] for i in [0,1,4,6,12,17,20,31]]
    results={}
    for defer in [False,True]:
        results[defer]=await execute(rows,[URL],f'{name}_N8_defer{int(defer)}',.25,2026112402,16,True,
            metadata=dict(evidence='CORRECTNESS_ONLY',manifest_SHA256=digest(path)),defer_raw=defer)
    strict=all(a['status']==b['status']=='MEASURED_ONLINE' and a['token_ids']==b['token_ids'] and
        a['usage']==b['usage'] for a,b in zip(results[False],results[True]))
    proof=dict(time=time.time(),status='PASS' if strict else 'FAILED',strict_same_geometry=strict,count=len(rows),
        scope='Client logging only. All actual model and E/P/D runtime paths remain the same.')
    write(ROOT/'analysis'/f'{name}_N8_correctness.json',proof)
    assert strict,'Stop the evidence control if model or usage results differ'


async def pilot(name,natural):
    path=resolve_manifest(None);rows=json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    counts=None if natural else load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    summary=[]
    for index,defer in enumerate([False,True,False]):
        run=f'{name}_arm{index}_defer{int(defer)}'
        results=await execute(rows,[URL],run,2.,2026112402,counts,
            metadata=dict(evidence='DIAGNOSTIC_C3',manifest_SHA256=digest(path),
                workers=WORKERS,processor_handoff='grid_only',quality_gate='PENDING',
                scope='One initialization. Immediate/deferred/immediate client evidence control. Not a token split arm.'),
            defer_raw=defer)
        assert len(results)==100 and all(row['status']=='MEASURED_ONLINE' for row in results)
        value=summarize(run)
        summary.append(value)
        print('EVIDENCE_CONTROL_ARM',run,{key:value[key] for key in
            ['TTFT_p95_ms','ITL_p99_ms','ITL_max_ms','TPOT_p95_ms','gate_attainment']},flush=True)
        del results
    write(ROOT/'analysis'/f'{name}_summary.json',dict(time=time.time(),rows=summary,
        evidence='DIAGNOSTIC_C3',independent_starts=1,confidence_interval='NOT_AVAILABLE',
        claim='This tests a client measurement-path artifact. It is not actual additional prefill work or token-split gain.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--natural',action='store_true');args=parser.parse_args()
    preserve();leases=readiness()
    archive=snapshot(args.name)
    plan=ROOT/'configs'/f'{args.name}_plan.json'
    assert not plan.exists()
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_CANDIDATE',workers=leases,
        source_archive_SHA256=archive['archive_SHA256'],one_factor='Client full SSE evidence serialization time',
        hypotheses=[
            'If completion raw serialization causes a receipt stall, post-drain full serialization reduces client ITL without changing engine output.',
            'If API or GPU timing dominates, the deferred policy does not restore the frozen all-ITL gate.',
            'If the first/last immediate arms differ materially, this single-start result is unstable.'],
        arms=['immediate','deferred','immediate'],QPS=2,count=100,
        output='Natural EOS' if args.natural else 'Per-request equal-output replay',
        fixed=['GPU group','model','processor','grid-only metadata handoff','arrival seed','order seed','SLO thresholds'],
        SLO=dict(every_ITL_ms=34,TPOT_ms=33,TTFT_ms=5000,absolute_decode_attainment_min=.95),
        caveat='Full chunks and prompt IDs remain in memory until drain. A compact completion journal preserves metrics if interrupted. '
            'This is not a server optimization or a token-split result.'))
    asyncio.run(correctness(args.name))
    asyncio.run(pilot(args.name,args.natural))
    preserve()


if __name__=='__main__':main()
