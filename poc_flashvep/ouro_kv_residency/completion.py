"""Check the original index, unchanged runtime, and all eight restored burners."""
import os
import subprocess
import time
from pathlib import Path
from .common import ROOT,REPO,FORK,write
from .resources import burners,processes,restore

def main():
    expected='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9'
    index=subprocess.check_output('git diff --cached --binary | sha256sum',shell=True,cwd=REPO,text=True).split()[0]
    assert index==expected
    assert not subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    restore(list(range(8)))
    owned=burners();active=processes()
    assert set(owned)==set(range(8))
    assert {r['gpu'] for r in active}==set(range(8))
    for row in active:
        proc=Path('/proc')/str(row['pid'])
        assert proc.stat().st_uid==os.getuid() and os.getsid(row['pid'])==owned[row['gpu']]
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    write(ROOT/'completion_audit.json',dict(wall_s=time.time(),unrelated_index_sha256=index,
        runtime_tracked_diff_empty=True,owned_burner_sessions=owned,active=active,
        observed_GPU_stats=stats,all_eight_burners_restored=True))
    print(stats)

if __name__=='__main__':main()
