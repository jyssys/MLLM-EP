"""Wait for the owned diagnostic lease, then run the registered attention probe."""
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write
from .dispatch import completed
from .resources import restore

def owned_controllers_live():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            if proc.stat().st_uid==os.getuid() and any(name in words for name in (
                'poc_flashvep.ouro_adaptive_chunk_loop.controller','poc_flashvep.ouro_adaptive_chunk_loop.dispatch')):
                return True
        except (OSError,UnicodeError):
            continue
    return False

def main():
    until=time.monotonic()+3*3600
    while owned_controllers_live():
        assert time.monotonic()<until,'Owned diagnostic controller did not finish'
        write(ROOT/'state'/'attention_followup.json',dict(status='WAITING_FOR_P1_LEASE',wall_s=time.time()))
        time.sleep(2)
    p1=__import__('json').loads((ROOT/'p1_diagnostic_plan.json').read_text())
    assert all(completed(j) for j in p1['jobs']),'Review incomplete Phase1 jobs before this probe'
    try:
        result=subprocess.run([PYTHON,'-u','-m','poc_flashvep.ouro_adaptive_chunk_loop.dispatch',
            '--manifest',str(ROOT/'p1_attention_probe_plan.json'),'--manage-burners'],cwd=REPO)
        assert result.returncode==0,'Attention probe failed. Preserve evidence.'
        write(ROOT/'state'/'attention_followup.json',dict(status='PROBE_COMPLETE_REVIEW_REQUIRED',wall_s=time.time()))
    finally:
        restore(list(range(8)))

if __name__=='__main__':
    main()
