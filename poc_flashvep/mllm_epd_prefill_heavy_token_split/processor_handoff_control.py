"""Bounded same-group correctness and pilot controls for processed prompts."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time
import urllib.request

from .common import ROOT,digest,preserve,write
from .client import execute,resolve_manifest,load_replay_counts
from .metrics import summarize
from .source_snapshot import snapshot


URLS={'none':'http://127.0.0.1:18843','full':'http://127.0.0.1:18844','grid_only':'http://127.0.0.1:18845'}


def ready():
    until=time.time()+1500
    while True:
        passed=True
        for name in ['metadata_B2_E4','metadata_B2_P5','metadata_B2_D6']:
            path=ROOT/'state'/f'server_{name}.json'
            if not path.exists():passed=False;continue
            row=json.loads(path.read_text())
            assert not row.get('finished'), 'Owned server terminated: '+name
            process=Path('/proc',str(row['pid']))
            assert process.exists() and process.stat().st_uid==os.getuid(), 'Server process missing or ownership differs'
            if not row.get('ready'):passed=False;continue
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{row["port"]}/health',timeout=2) as response:
                    passed=passed and response.status==200
            except OSError:passed=False
        if passed:return
        if time.time()>until:raise TimeoutError('Readiness observation expired. Inspect the same leases before any restart.')
        time.sleep(2)


async def correctness(name,indices,output_count=16):
    path=ROOT/'data/calibration_manifest_template_v2.json'
    manifest=json.loads(path.read_text());rows=[manifest['requests'][i] for i in indices]
    results={}
    for mode,url in URLS.items():
        run=f'{name}_{mode}'
        results[mode]=await execute(rows,[url],run,.25,2026112402,output_count,True,
            metadata=dict(evidence='CORRECTNESS_ONLY',manifest=str(path),manifest_SHA256=digest(path),
                processor_handoff=mode,server_names=['metadata_B2_E4','metadata_B2_P5','metadata_B2_D6']))
        assert len(results[mode])==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in results[mode]), mode
    baseline={r['request_id']:r for r in results['none']};comparison=[]
    for mode,values in results.items():
        for row in values:
            reference=baseline[row['request_id']]
            comparison.append(dict(mode=mode,request_id=row['request_id'],
                prompt_match=row['prompt_SHA256']==reference['prompt_SHA256'],
                actual_prompt_tokens=row['usage']['prompt_tokens'],expected_prompt_tokens=reference['usage']['prompt_tokens'],
                strict_tokens=row['token_ids']==reference['token_ids']))
    proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY',indices=indices,comparison=comparison,
        passed=all(r['prompt_match'] and r['strict_tokens'] and r['actual_prompt_tokens']==r['expected_prompt_tokens'] for r in comparison),
        scope='Same GPU4/5/6 servers and serial execution geometry. Compare each reuse path with the unmodified no-handoff path. '
            'This does not waive cross-geometry encoder or full task-quality gates.')
    write(ROOT/'analysis'/f'{name}.json',proof)
    assert proof['passed'], 'Processed-prompt output mismatch. Do not proceed to timing.'
    print('PROCESSOR_HANDOFF_CORRECTNESS_PASS',name,len(rows),flush=True)


async def pilot(name,mode,qps):
    assert mode in URLS
    gate=json.loads((ROOT/'analysis/processor_handoff_B2_N8_natural_v0.json').read_text())
    assert gate['passed'], 'N8 correctness gate did not pass'
    path=resolve_manifest(None);rows=json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    raw=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    counts=load_replay_counts(raw)
    result=await execute(rows,[URLS[mode]],name,qps,2026112402,counts,
        metadata=dict(evidence='DIAGNOSTIC_CONTROL',manifest=str(path),manifest_SHA256=digest(path),
            replay_reference_SHA256=digest(raw),processor_handoff=mode,
            quality_gate='PENDING',scope='Same-group one-factor common-output online runtime control. Not a new token split method.'))
    assert len(result)==100 and all(r['status']=='MEASURED_ONLINE' for r in result)
    value=summarize(name)
    print('PROCESSOR_HANDOFF_PILOT',name,{k:value[k] for k in ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','gate_attainment']},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['correctness','pilot'])
    parser.add_argument('--name');parser.add_argument('--mode',choices=list(URLS),default='grid_only')
    parser.add_argument('--qps',type=float,default=2)
    parser.add_argument('--natural',action='store_true',help='Full Natural EOS correctness, not fixed-output timing')
    args=parser.parse_args();preserve()
    archive=snapshot('processor_handoff_'+args.phase)
    print('PROCESSOR_HANDOFF_SOURCE_ARCHIVE',archive['archive_SHA256'],flush=True)
    ready();print('PROCESSOR_HANDOFF_SERVERS_READY',flush=True)
    if args.phase=='correctness':
        suffix='natural_v0' if args.natural else 'v0'
        output_count=None if args.natural else 16
        asyncio.run(correctness('processor_handoff_B2_N1_'+suffix,[0],output_count))
        asyncio.run(correctness('processor_handoff_B2_N8_'+suffix,[0,1,4,6,12,17,20,31],output_count))
    else:
        assert args.name
        asyncio.run(pilot(args.name,args.mode,args.qps))
    preserve()


if __name__=='__main__':main()
