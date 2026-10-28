"""Persistent GPU leases. A completed job is never executed twice."""
import argparse
import json
import os
import subprocess
import sys
import time
from .common import ROOT,REPO,PYTHON,environment,write,record_command,setup
from . import resources

def _execute(manifest):
 setup();jobs=json.loads(open(manifest).read());running={}
 resources.base.stop(list(range(8)))
 try:
  while jobs or running:
   busy={g for j,p,f in running.values() for g in j['gpus']}
   for j in list(jobs):
    status=ROOT/'raw'/j['name']/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='COMPLETE':jobs.remove(j);continue
    if set(j['gpus'])&busy:continue
    if any(other.get('quiet') for other,p,f in running.values()):continue
    if j.get('quiet') and running:continue
    if any(row['gpu'] in j['gpus'] for row in resources.base.processes()):continue
    log=(ROOT/'logs'/f"{j['name']}.log").open('a')
    command=[j.get('python',PYTHON),'-m',j.get('module','poc_flashvep.looped_pd_abc.worker'),*j['args']]
    env=environment(j['gpus']);env.update(j.get('env',{}))
    env['ABC_LANE']='isolated_quiet' if j.get('quiet') else 'parallel_discovery'
    p=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record_command(j['name'],command,j['gpus'],p.pid)
    running[j['name']]=(j,p,log);busy.update(j['gpus']);jobs.remove(j)
   for name,(j,p,f) in list(running.items()):
    if p.poll() is not None:
     f.close();del running[name]
     write(ROOT/'state'/('exit_'+name+'.json'),dict(returncode=p.returncode,time=time.time()))
   write(ROOT/'state/dispatcher.json',dict(pid=os.getpid(),running={n:p.pid for n,(j,p,f) in running.items()},pending=[j['name'] for j in jobs],time=time.time()))
   if jobs or running:time.sleep(2)
 finally:resources.base.restore(list(range(8)))

def execute(manifest):
 import fcntl
 setup()
 with (ROOT/'state/dispatcher.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  # A controller started before the locking update can still own the pool.
  old=ROOT/'state/dispatcher.json'
  previous=json.loads(old.read_text()).get('pid') if old.exists() else None
  if previous and previous!=os.getpid():
   proc=__import__('pathlib').Path(f'/proc/{previous}')
   while proc.exists():
    try:
     words=(proc/'cmdline').read_bytes()
     if proc.stat().st_uid!=os.getuid() or b'poc_flashvep.looped_pd_abc.dispatch' not in words:break
    except (FileNotFoundError,ProcessLookupError):break
    time.sleep(2)
  _execute(manifest)

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--detach',action='store_true');a=p.parse_args()
 if a.detach:
  setup()
  with (ROOT/'logs/dispatcher.log').open('a') as f:
   child=subprocess.Popen([sys.executable,'-m','poc_flashvep.looped_pd_abc.dispatch',a.manifest],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  print('DISPATCHER',child.pid)
 else:execute(a.manifest)

if __name__=='__main__':main()
