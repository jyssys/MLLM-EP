"""Inspect native spawn and title functions without a model or CUDA context."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from poc_ouro_parallelism_deep_discovery.contract import ROOT, write


CHAIN_CODE = r'''
import json, os, sys
from pathlib import Path
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
configure_native_imports()
import torch
from vllm.utils.system_utils import get_mp_context, set_process_title
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import inspect_process_leases
set_process_title(('Frontend', 'EngineCore', 'Worker_TP0')[depth])
expected = 'CPU_SPAWN_CHAIN_GUARD'
entries = Path('/proc/self/environ').read_bytes().split(b'\0')
row = dict(role=('frontend','engine_core','rank')[depth], pid=os.getpid(), ppid=os.getppid(),
 logical_guard_matches=os.environ.get('LOOPED_KV_OUTPUT')==expected,
 proc_guard_matches=('LOOPED_KV_OUTPUT='+expected).encode() in entries,
 cuda_initialized=torch.cuda.is_initialized(), mp_method=get_mp_context().get_start_method())
try:
 inspect_process_leases([os.getpid()],os.getpgid(0),os.getsid(0),expected,ancestor_pid=os.getpid())
 row['strict_lease_pass']=True
except RuntimeError as error:
 row['strict_lease_pass']=False
 row['error']=str(error)
rows=[row]
if depth < 2:
 ctx=get_mp_context()
 parent, child=ctx.Pipe()
 process=ctx.Process(target=exec,args=(code,dict(code=code,depth=depth+1,reply=child)))
 process.start()
 child.close()
 assert parent.poll(30), 'CPU child timed out'
 rows.extend(parent.recv())
 parent.close()
 process.join(30)
 assert process.exitcode==0, process.exitcode
assert not torch.cuda.is_initialized()
if depth:
 reply.send(rows)
 reply.close()
else:
 print(json.dumps(rows))
 assert all(row['strict_lease_pass'] for row in rows)
 assert not any(row['cuda_initialized'] for row in rows)
'''


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        raise ValueError('Require a new artifact path inside this campaign.')
    native = Path(sys.executable).parents[1] / 'lib/python3.14/site-packages'
    paths = [Path(__file__).resolve(), ROOT / 'runtime.py', ROOT / 'tp2_dp2_cohort.py',
             native / 'vllm/utils/system_utils.py',
             native / 'vllm/v1/executor/multiproc_executor.py',
             native / 'vllm/v1/engine/utils.py']
    sources = {str(path): sha(path) for path in paths}
    start = utc_now()
    environment = dict(os.environ, LOOPED_KV_OUTPUT='CPU_SPAWN_CHAIN_GUARD',
                       CUDA_VISIBLE_DEVICES='', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                       VLLM_LOGGING_LEVEL='ERROR')
    script = 'depth=0\ncode=' + repr(CHAIN_CODE) + '\nexec(code)'
    result = subprocess.run([sys.executable, '-c', script], env=environment,
                            capture_output=True, text=True, timeout=90)
    end = utc_now()
    rows = json.loads(result.stdout) if result.returncode == 0 else None
    passed = result.returncode == 0 and len(rows) == 3 and all(
        row['strict_lease_pass'] and row['logical_guard_matches'] and row['proc_guard_matches']
        and row['mp_method'] == 'spawn' and not row['cuda_initialized'] for row in rows)
    record = dict(start_utc=start, end_utc=end, command=[sys.executable, '-m',
        'poc_ouro_parallelism_deep_discovery.scripts.process_lease_spawn_chain_cpu_v1',
        '--output', str(output)], subprocess_returncode=result.returncode, rows=rows,
        source_sha256_before_run=sources, stdout=result.stdout,
        stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest(),
        stderr_line_count=len(result.stderr.splitlines()),
        inherited_SPT_NOENV_present='SPT_NOENV' in os.environ,
        seam_passed=passed, actual_native_startup_reproduced=False,
        scope='Actual native spawn/title and strict self lease. No WorkerProc constructor, model, or CUDA context.',
        limitation='A passing chain does not diagnose the full native TP2 startup failure.')
    write(output, record)
    print(json.dumps(dict(seam_passed=passed, rows=rows, output=str(output))))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
