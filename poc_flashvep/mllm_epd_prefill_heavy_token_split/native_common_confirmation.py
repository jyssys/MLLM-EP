"""Two new independent starts of the frozen common Native compiler policy."""
import json
import os
from pathlib import Path
import subprocess
import time

from .common import ROOT, REPO, PYTHON, digest, experiment_deadline, preserve, write


def wait_closed(tag):
    while True:
        pending = []
        for role, gpu in [('E', 4), ('P', 5), ('D', 6)]:
            path = ROOT / 'state' / f'server_{tag}_{role}{gpu}.json'
            lease = json.loads(path.read_text())
            process = Path('/proc', str(lease['guardian']))
            if process.exists():
                assert process.stat().st_uid == os.getuid()
                command = (process / 'cmdline').read_bytes()
                assert b'mllm_epd_prefill_heavy_token_split.servers' in command and tag.encode() in command
                pending.append(str(process))
            else:
                assert lease['finished'], 'Missing guardian without terminal resource record'
        if not pending:
            break
        assert time.time() < experiment_deadline()
        time.sleep(1)
    result = ROOT / 'analysis' / f'{tag}_pilot_summary.json'
    record = json.loads(result.read_text())
    assert len(record['results']) == 3
    assert all(row['primary_execution_valid'] and row['completed_count'] == 100 for row in record['results'])
    return dict(tag=tag, summary_SHA256=digest(result))


def main():
    preserve()
    plan = ROOT / 'configs/common_native_confirmation_C32_v0.json'
    assert not plan.exists()
    write(plan, dict(time=time.time(), status='FROZEN_BEFORE_TWO_FOLLOWUP_STARTS',
        prior_start='ec_COMMON_C32', followup_starts=['ec_COMMON_C32R1', 'ec_COMMON_C32R2'],
        QPS=2, manifest=str(ROOT / 'data/main100_manifest_template_v2.json'),
        manifest_SHA256=digest(ROOT / 'data/main100_manifest_template_v2.json'),
        policy='Same common compiled Native policy. No token-split adapter or extra admission cap.',
        independence='Retire E/P/D, restore approved burners, then initialize a new ownership epoch per start.',
        repetitions='One pilot plus two followup starts. No QPS, compiler, SLO or output-policy tuning.',
        correctness='Repeat N1 and serial N8 state gates in each new start before timing.',
        output='Natural EOS and per-request equal-output replay remain separate.',
        purpose='Quantify the explicit common numerical policy before candidate comparisons.',
        quality='PENDING. These repetitions do not authorize or replace official external answer extraction.'))
    complete = []
    previous = 'ec_COMMON_C32'
    for tag in ['ec_COMMON_C32R1', 'ec_COMMON_C32R2']:
        complete.append(wait_closed(previous))
        assert time.time() < experiment_deadline()
        log = ROOT / 'logs' / f'{tag}_v0.log'
        command = [str(PYTHON), '-m', 'poc_flashvep.mllm_epd_prefill_heavy_token_split.native_common_control',
            '--tag', tag]
        with log.open('xb') as stream:
            process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            write(ROOT / 'state' / f'{tag}_controller.json', dict(time=time.time(), pid=process.pid,
                command=command, status='RUNNING'))
            while process.poll() is None:
                time.sleep(1)
            assert process.returncode == 0, 'Inspect this failed start. Do not restart it under another label.'
        previous = tag
    complete.append(wait_closed(previous))
    write(ROOT / 'state/common_native_confirmation_C32_complete.json', dict(time=time.time(),
        status='COMPLETE', independent_initializations=3, completed=complete,
        quality='PENDING', resource_closeout='Per-start closure only. Overall campaign remains active.'))
    preserve()


if __name__ == '__main__':
    main()
