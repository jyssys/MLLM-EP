"""Insert three owned-pair interference witnesses at a completed-job boundary."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest
from .run import save
from .metrics import summarize
from .service_accounting import analyze


def main():
    parent=3764359
    path=Path('/proc',str(parent),'cmdline')
    expected=b'poc_flashvep.dense_ar_longbench400_pd.queue\0motivation400_discovery_v1.json'
    assert path.stat().st_uid==os.getuid() and expected in path.read_bytes()
    receipt=ROOT/'state/main_queue_interference_pause_v1.json';assert not receipt.exists()
    baseline='PairInterference_Serial0_v1'
    old=ROOT/'raw'/baseline
    capsule=ROOT/'references/source_capsules/pair_interference_serial_control_v1.zip'
    if not capsule.exists():shutil.copyfile(old/'source.zip',capsule)
    original=json.loads((old/'configuration.json').read_text());jobs=[]
    for index in range(3):
        cfg=dict(original,name=f'PairInterference_Concurrent0_Start{index}_v1',
            source_capsule=capsule.name,source_capsule_SHA256=digest(capsule),
            phase='PAIRED_SOURCE_CAPSULE_PARALLEL_INTERFERENCE_WITNESS')
        jobs.append(save(cfg))
    write(receipt,dict(pid=parent,status='REQUESTED',time=time.time(),jobs=jobs,
        scope='Pause only the owned queue parent. Its active GPU job drains normally. Resume the parent in finally.'))
    os.kill(parent,signal.SIGSTOP)
    try:
        results=[]
        reference_metric=summarize(baseline);reference_service=analyze(baseline)
        for name in jobs:
            snapshots={p.name:json.loads(p.read_text()) for p in (ROOT/'state').glob('pair_lease_*.json')}
            code=subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.run',name],cwd=REPO).returncode
            assert code==0,('Interference witness failed',name)
            metric=summarize(name);service=analyze(name)
            ratios={key:metric[key]/reference_metric[key] for key in ('TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms')}
            ratios.update({key:service['means'][key]/reference_service['means'][key] for key in ('P_submit_to_ready_ms','D_submit_to_finish_ms')})
            ratios['throughput_slowdown']=reference_metric['generated_output_tokens_per_second']/metric['generated_output_tokens_per_second']
            results.append(dict(name=name,ratios=ratios,passed=max(ratios.values())<=1.05,
                other_pair_lease_snapshot_before_wait=snapshots))
            write(ROOT/'analysis/pair_interference_preflight_v1.json',dict(baseline=baseline,results=results,
                passed=len(results)==3 and all(r['passed'] for r in results),limit=1.05,
                evidence='MEASURED_RESOURCE_PREFLIGHT',
                scope='Same GPU0/1 and immutable source capsule. Three parallel witnesses compare with one prior serial start. '
                    'This is not a three-start versus three-start performance CI. Verify actual other-pair activity during each window. '
                    'A failed threshold requires an isolated drift control before causal attribution.'))
    finally:
        if path.exists():
            assert expected in path.read_bytes();os.kill(parent,signal.SIGCONT)
        write(receipt,dict(pid=parent,status='RESUMED',time=time.time(),jobs=jobs))


if __name__=='__main__':main()
