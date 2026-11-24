"""Fresh, unchanged E/P/D initialization for the missing q3 activity control."""
import asyncio
import json
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace
import urllib.request

from .common import ROOT,PYTHON,REPO,preserve,write
from .client import execute,load_replay_counts,resolve_manifest
from .profile_control import run as profile_run
from .source_snapshot import snapshot


def main():
    preserve();archive=snapshot('metadata_B3_independent_q3_profile')
    old=['metadata_B2_E4','metadata_B2_P5','metadata_B2_D6']
    until=time.time()+180
    while any(Path('/proc',str(json.loads((ROOT/'state'/f'server_{name}.json').read_text())['guardian'])).exists() for name in old):
        assert all((ROOT/'state'/f'stop_server_{name}').exists() for name in old)
        if time.time()>until:raise TimeoutError('Old guard closeout is still live. Inspect the same guards before another launch.')
        time.sleep(1)
    names=[]
    for role,gpu,port,side in [('encoder',4,18840,19840),('prefill',5,18841,19841),('decode',6,18842,19842)]:
        name=f'metadata_B3_{role[0].upper()}{gpu}';names.append(name)
        command=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',name,'--gpu',str(gpu),'--role',role,'--port',str(port),'--side-port',str(side),
            '--ec-path','/dev/shm/mllm_epd_20261124_metadata_B3','--observe']
        if role=='encoder':command+=['--encoder-native-ops','--compiled-encoder','--encoder-sync-scheduling']
        subprocess.run(command,cwd=REPO,check=True)
    write(ROOT/'configs/metadata_B3_independent_q3_profile.json',dict(time=time.time(),servers=names,
        source_archive_SHA256=archive['archive_SHA256'],QPS=3,policy_change=False,
        purpose='Fresh workers are required for a second CUPTI capture. Do not remove the one-profile guard.',
        evidence='DIAGNOSTIC until observer and model-quality gates pass'))
    until=time.time()+1500
    while True:
        ready=True
        for name in names:
            path=ROOT/'state'/f'server_{name}.json'
            if not path.exists():ready=False;continue
            lease=json.loads(path.read_text())
            assert not lease.get('finished'),name+' terminated'
            proc=Path('/proc',str(lease['pid']))
            assert proc.exists() and proc.stat().st_uid==os.getuid()
            if not lease.get('ready'):ready=False;continue
            with urllib.request.urlopen(f'http://127.0.0.1:{lease["port"]}/health',timeout=3) as response:
                ready=ready and response.status==200
        if ready:break
        if time.time()>until:raise TimeoutError('Readiness observation expired. Inspect these same live leases before any restart.')
        time.sleep(2)
    print('INDEPENDENT_B3_SERVERS_READY',flush=True)
    calibration=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests'][:1]
    current=asyncio.run(execute(calibration,['http://127.0.0.1:18845'],'metadata_B3_N1_v0',.25,2026112402,16,True,
        metadata=dict(evidence='CORRECTNESS_ONLY')))
    reference=json.loads((ROOT/'raw/processor_handoff_B2_N1_v0_grid_only_requests.jsonl').read_text().splitlines()[0])
    assert current[0]['status']=='MEASURED_ONLINE' and current[0]['token_ids']==reference['token_ids']
    print('INDEPENDENT_B3_N1_PASS',flush=True)
    manifest=resolve_manifest(None);rows=json.loads(manifest.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    raw=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    warm=asyncio.run(execute(rows,['http://127.0.0.1:18845'],'metadata_B3_q3_warmup_v0',3,2026112402,load_replay_counts(raw),
        metadata=dict(evidence='DIAGNOSTIC_WARMUP')))
    assert len(warm)==100 and all(r['status']=='MEASURED_ONLINE' for r in warm)
    print('INDEPENDENT_B3_WARMUP_PASS',flush=True)
    config=SimpleNamespace(manifest=manifest,replay_from=raw,server_names=','.join(names),
        order_seed=2026112404,qps=3,seed=2026112402,name='metadata_B3_grid_main100_CPC_q3_v0',
        url='http://127.0.0.1:18845',worker_urls='http://127.0.0.1:18840,http://127.0.0.1:18841,http://127.0.0.1:18842',
        observer_kind='profile',observe_client_host=False,freeze_client_static_heap=False)
    asyncio.run(profile_run(config));preserve()


if __name__=='__main__':main()
