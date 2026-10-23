"""Sequential evidence stages. No GPU stage silently follows a failed stage."""
import fcntl
import os
import subprocess
import sys
import time
from .common import ROOT,REPO,write

def main():
    names=[a for a in sys.argv[1:] if a!='--wait']
    lock=(ROOT/'state'/'chain.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX if '--wait' in sys.argv else fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in names:
        manifest=ROOT/name
        assert manifest.parent==ROOT and manifest.exists()
        # Respect an already running dispatcher, including its burner restoration.
        while True:
            probe=(ROOT/'state'/'dispatcher.lock').open('a')
            try:
                fcntl.flock(probe.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                probe.close();break
            except BlockingIOError:
                probe.close();time.sleep(2)
        write(ROOT/'research_state.json',dict(stage=name,pid=os.getpid(),wall_s=time.time(),status='RUNNING',next=names[names.index(name)+1:]))
        code=subprocess.call(['/home/esjung/anaconda3/bin/python','-u','-m','poc_flashvep.ouro_elb.dispatch',
            '--manifest',str(manifest),'--manage-burners'],cwd=REPO)
        write(ROOT/'research_state.json',dict(stage=name,pid=os.getpid(),wall_s=time.time(),status='COMPLETE' if code==0 else 'REVIEW_REQUIRED',exit_code=code))
        if code:raise SystemExit(code)

if __name__=='__main__':main()
