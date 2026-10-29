"""Insert a reference audit between complete jobs. Never pause an inference worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,REPO,write,environment,record_command
from . import resources

def identity(pid):
 proc=Path('/proc')/str(pid)
 if not proc.exists() or proc.stat().st_uid!=os.getuid():return None
 command=(proc/'cmdline').read_bytes()
 if b'poc_flashvep.looped_pd_abc.run_benchmarks' not in command:return None
 return (proc/'stat').read_text().split()[21]

def main():
 dest=ROOT/'state/quality_interposition.json';state=json.loads((ROOT/'state/main_campaign.json').read_text())
 pid=state['pid'];token=identity(pid);paused=False;child=None
 if not token or state.get('phase')!='serving':raise RuntimeError('Expected the verified owned serving controller')
 def interrupted(number,frame):raise SystemExit('Interposition interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  os.kill(pid,signal.SIGSTOP);paused=True
  write(dest,dict(status='WAITING_FOR_CURRENT_INFERENCE_TO_COMPLETE',pid=os.getpid(),controller=pid,time=time.time(),
   rule='Only the launcher is paused. The actual inference processes continue without interruption.'))
  deadline=time.monotonic()+2400
  while True:
   active=resources.base.processes()
   if not active:break
   for row in active:
    process=Path('/proc')/str(row['pid']);session=os.getsid(row['pid']);leader=Path('/proc')/str(session)
    if process.stat().st_uid!=os.getuid() or not leader.exists():raise RuntimeError('Unowned GPU process. No process was signaled.')
    command=(leader/'cmdline').read_bytes()
    if b'poc_flashvep.looped_pd_abc.worker' not in command:raise RuntimeError('Unexpected GPU workload. No GPU process was signaled.')
   if time.monotonic()>deadline:raise TimeoutError('Current inference remains active. Resume the launcher without an audit.')
   time.sleep(2)
  name='ouro_reference_gate_audit';folder=ROOT/'raw'/name
  if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text()).get('status')=='COMPLETE':return
  write(dest,dict(status='OFFICIAL_REFERENCE_AUDIT',pid=os.getpid(),controller=pid,time=time.time(),
   physical_GPU=6,benchmark_jobs_active=False))
  env=environment([6]);env['ABC_DIAGNOSTIC_BURN_OTHERS']='1';env['ABC_LANE']='reference_diagnostic_not_serving_timing'
  command=['/home/esjung/.venvs/looped-pd-abc-hf457/bin/python','-m','poc_flashvep.looped_pd_abc.ouro_gate_audit','--name',name]
  with (ROOT/'logs'/f'{name}.log').open('a') as log:
   child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   record_command(name,command,[6],child.pid)
   try:code=child.wait(timeout=1800)
   except subprocess.TimeoutExpired:
    child.terminate()
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    raise TimeoutError('Owned reference audit exceeded its 1800-second diagnostic limit')
  write(dest,dict(status='AUDIT_FINISHED_CLEANUP',returncode=code,pid=os.getpid(),controller=pid,time=time.time()))
 except BaseException as error:
  write(dest,dict(status='INTERPOSITION_FAILED_LAUNCHER_WILL_RESUME',error=repr(error),pid=os.getpid(),controller=pid,time=time.time()))
  raise
 finally:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  # The audit may have restored owned burners on the other seven devices.
  # Remove only those verified sessions before the next clean benchmark starts.
  try:resources.base.stop(list(range(8)))
  finally:
   if paused and identity(pid)==token:os.kill(pid,signal.SIGCONT)
   write(ROOT/'state/quality_interposition_release.json',dict(controller=pid,resumed=identity(pid)==token,time=time.time(),
    note='No inference worker received a pause or termination signal.'))

if __name__=='__main__':main()
