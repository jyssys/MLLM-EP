"""Preserve a safe closeout if the interactive agent loses capacity."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from .common import *

MODULE='poc_flashvep.adaptive_loop_token_pd_2gpu.'

def live_controller():
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            words=(path/'cmdline').read_bytes().split(b'\0')
            if path.stat().st_uid==os.getuid() and (MODULE+'campaign').encode() in words and b'run' in words:return True
        except OSError:pass
    return False

def stop_owned_at_deadline():
    ledger=ROOT/'commands.jsonl'
    for line in ledger.read_text().splitlines() if ledger.exists() else []:
        record=json.loads(line);pid=record['pid'];path=Path('/proc')/str(pid)
        try:
            words=(path/'cmdline').read_bytes().decode().split('\0')
            if path.stat().st_uid!=os.getuid() or os.getsid(pid)!=pid:continue
            if MODULE+'worker' not in words or not any(str(ROOT/'configs') in x for x in words):continue
            if words[:len(record['command'])]!=record['command']:continue
            status=ROOT/'raw'/record['name']/'status.json'
            previous=json.loads(status.read_text()) if status.exists() else {}
            if previous.get('status') in ('COMPLETE','FAILED'):continue
            append(ROOT/'state/deadline_termination.jsonl',dict(name=record['name'],pid=pid,time=time.time(),remaining_s=remaining(),previous_status=previous))
            os.killpg(pid,signal.SIGTERM)
            write(status,dict(status='TERMINATED_TIME_BUDGET',time=time.time(),previous_status=previous))
        except (OSError,UnicodeError):pass

def run():
    setup()
    while True:
        status=ROOT/'state/controller.json'
        value=json.loads(status.read_text()) if status.exists() else {}
        active=live_controller()
        if not active and value.get('status')=='PRIMARY_FINISHED_NEEDS_REVIEW':break
        if remaining()<900:
            stop_owned_at_deadline()
            while live_controller() and remaining()>600:time.sleep(2)
            if live_controller():
                for path in Path('/proc').iterdir():
                    if not path.name.isdecimal():continue
                    try:
                        pid=int(path.name);words=(path/'cmdline').read_bytes().split(b'\0')
                        if path.stat().st_uid==os.getuid() and os.getsid(pid)==pid and (MODULE+'campaign').encode() in words and b'run' in words:
                            os.killpg(pid,signal.SIGTERM)
                    except OSError:pass
                time.sleep(3)
            break
        if not active and remaining()<3600:break
        time.sleep(10)
    from .resources import base as resources
    resources.restore(list(range(8)))
    write(ROOT/'state/returned_resources.json',dict(time=time.time(),owned_burners=resources.burners(),
        active=resources.processes(),scope='Owned burner sessions and actual GPU process query at closeout.'))
    from .publication import publish
    try:
        publish()
        write(ROOT/'state/auto_publication_outcome.json',dict(status='COMMIT_AND_PUSH_CONFIRMED',time=time.time(),elapsed_hours=(time.time()-START_EPOCH)/3600))
    except Exception as error:
        write(ROOT/'state/auto_publication_outcome.json',dict(status='PUBLICATION_FAILED',error=repr(error),time=time.time(),elapsed_hours=(time.time()-START_EPOCH)/3600))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run']);a=p.parse_args();setup()
    if a.action=='run':return run()
    executable=shutil.which('apply_patch')
    if not executable:raise RuntimeError('Cannot prepare safe automated report edits without apply_patch')
    env=dict(os.environ,LTP_APPLY_PATCH=executable)
    with (ROOT/'logs/auto_publication.log').open('a') as handle:
        child=subprocess.Popen([str(PYTHON),'-m',MODULE+'auto_closeout','run'],cwd=REPO,env=env,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/auto_closeout_supervisor.json',dict(pid=child.pid,time=time.time()))
    print('DETACHED_CLOSEOUT_WATCHER',child.pid)

if __name__=='__main__':main()
