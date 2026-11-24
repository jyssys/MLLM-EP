"""Run one bounded host-scope diagnostic after the owned decoder is ready."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request

from .common import ROOT, digest, preserve, write
from .client import execute, resolve_manifest, load_replay_counts
from .profile_control import run as profile_run
from .source_snapshot import snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='native_main100_hostscope_CPC_q2_v0')
    parser.add_argument('--decoder',default='native_timeline_D15_host_scopes')
    args=parser.parse_args()
    preserve()
    assert not (ROOT/'configs'/f'{args.name}.json').exists(), 'Do not restart an existing measurement'
    archived=snapshot(args.name+'_controller')
    wait_deadline=time.time()+1200
    while True:
        lease=json.loads((ROOT/'state'/f'server_{args.decoder}.json').read_text())
        assert not lease.get('finished'), 'The owned decoder is terminal'
        process=Path('/proc')/str(lease['pid'])
        assert process.exists() and process.stat().st_uid==os.getuid(), 'Decoder process is missing or ownership differs'
        if lease.get('ready'):
            with urllib.request.urlopen('http://127.0.0.1:18432/health',timeout=3) as response:
                if response.status==200:break
        if time.time()>wait_deadline:
            raise TimeoutError('Readiness observation expired. Inspect this same lease before any restart.')
        time.sleep(2)
    print('HOST_SCOPE_DECODER_READY',flush=True)
    calibration=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests'][:1]
    n1_name=args.name+'_N1'
    result=asyncio.run(execute(calibration,['http://127.0.0.1:18433'],n1_name,.25,2026112402,16,True,
        metadata=dict(evidence='CORRECTNESS_ONLY',source_archive_SHA256=archived['archive_SHA256'])))
    baseline=[json.loads(s) for s in (ROOT/'raw/native_timeline_N1_passthrough_v0_requests.jsonl').read_text().splitlines()]
    proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY',request_id=result[0]['request_id'],
        prompt_match=result[0]['prompt_SHA256']==baseline[0]['prompt_SHA256'],
        strict_token_match=result[0]['token_ids']==baseline[0]['token_ids'],
        source_SHA256=digest(__file__),decoder_lease=lease,
        scope='Single same-geometry request. This does not qualify Main100 task quality.')
    write(ROOT/'analysis'/f'{n1_name}.json',proof)
    assert proof['prompt_match'] and proof['strict_token_match'], 'Host-scope N1 output changed'
    print('HOST_SCOPE_N1_PASS',flush=True)
    manifest=resolve_manifest(None)
    rows=json.loads(manifest.read_text())['requests']
    import random
    random.Random(2026112404).shuffle(rows)
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    counts=load_replay_counts(reference)
    warm=asyncio.run(execute(rows,['http://127.0.0.1:18433'],args.name+'_warmup',2,2026112402,counts,
        metadata=dict(evidence='DIAGNOSTIC_WARMUP',manifest_SHA256=digest(manifest))))
    assert len(warm)==100 and all(r['status']=='MEASURED_ONLINE' for r in warm)
    print('HOST_SCOPE_WARMUP_PASS',flush=True)
    config=SimpleNamespace(manifest=manifest,replay_from=reference,
        server_names='native_timeline_E13_sync,native_timeline_P10,'+args.decoder,
        order_seed=2026112404,qps=2,seed=2026112402,name=args.name,
        url='http://127.0.0.1:18433',worker_urls='http://127.0.0.1:18430,http://127.0.0.1:18431,http://127.0.0.1:18432',
        observer_kind='timeline',observe_client_host=True,freeze_client_static_heap=False)
    asyncio.run(profile_run(config))
    preserve()


if __name__=='__main__':main()
