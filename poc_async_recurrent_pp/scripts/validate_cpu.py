"""Preserve the CPU test result without recording the process environment."""
import subprocess
import sys
from poc_async_recurrent_pp.contract import *


def main():
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','poc_async_recurrent_pp/tests','-v'],
                          cwd=PROJECT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    write(ROOT/'outputs/validation/cpu_tests.json',dict(command=result.args,return_code=result.returncode,
        stdout=result.stdout,test_source_sha256={p.name:sha(p) for p in (ROOT/'tests').glob('*.py')}))
    print(result.stdout)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
