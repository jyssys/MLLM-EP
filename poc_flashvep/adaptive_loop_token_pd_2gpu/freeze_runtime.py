"""Freeze the execution overlay before the held-out serving matrix."""
import subprocess
from .common import *

def main():
    path=ROOT/'configs/frozen_execution_hashes.json'
    if path.exists():raise FileExistsError(path)
    files={str(SOURCE/name):digest(SOURCE/name) for name in ('common.py','worker.py','connector.py','policy.py')}
    write(path,dict(time=time.time(),hashes=files,
        repo_HEAD_at_freeze=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        interpretation='Execution-file hashes are authoritative. HEAD does not imply a clean user worktree.'))
    print('EXECUTION_OVERLAY_FROZEN',files)

if __name__=='__main__':main()
