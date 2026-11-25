"""Bind the final cold-source CPU suite to its actual tested source hashes."""
import argparse
import os
import re
import subprocess
from tempfile import TemporaryDirectory
import time

from .common import ROOT,REPO,SOURCE,PYTHON,digest,preserve,write
from .resource_closeout import require_final_closeout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);args=parser.parse_args()
    assert re.fullmatch('v[0-9]+',args.version);preserve();require_final_closeout()
    output=ROOT/'analysis'/f'final_cpu_regression_{args.version}.json'
    log=ROOT/'logs'/f'cpu_regression_final_{args.version}.log';assert not output.exists() and not log.exists()
    before={path.name:digest(path) for path in SOURCE.glob('*.py')};start=time.time()
    with TemporaryDirectory(prefix='mllm_epd_final_cpu_pycache_') as directory:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=directory)
        command=[str(PYTHON),'-B','-X','pycache_prefix='+directory,'-m','unittest','discover',
            '-s','poc_flashvep/mllm_epd_prefill_heavy_token_split','-t','.']
        with log.open('xb') as stream:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
    after={path.name:digest(path) for path in SOURCE.glob('*.py')}
    text=log.read_text();match=re.search(r'Ran (\d+) tests',text)
    passed=result.returncode==0 and before==after and match is not None and text.rstrip().endswith('OK')
    write(output,dict(begin=start,end=time.time(),passed=passed,returncode=result.returncode,
        tests=int(match[1]) if match else None,command=command,log=str(log.relative_to(REPO)),log_SHA256=digest(log),
        source_SHA256=before,source_unchanged_during_tests=before==after,
        scope='Actual CPU suite with a fresh pycache lookup path and bytecode writes disabled. '
            'No existing source bytecode cache supplies the tested campaign modules. GPU correctness is separate.'))
    preserve();assert passed,'Preserve failed CPU audit and inspect the actual log'
    print('FINAL_COLD_SOURCE_CPU_SUITE_PASS',int(match[1]),flush=True)


if __name__=='__main__':main()
