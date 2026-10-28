"""Reap verified owned FAILED sessions. Never signal a healthy or unrelated job."""
import json
import os
from pathlib import Path
import signal
import time
from .common import ROOT,write,append

def verified(pid,name):
 proc=Path('/proc')/str(pid)
 if not proc.exists():return False
 try:
  words=(proc/'cmdline').read_bytes()
  return (proc.stat().st_uid==os.getuid() and os.getpgid(pid)==pid and os.getsid(pid)==pid
   and b'poc_flashvep.looped_pd_abc.' in words and name.encode() in words)
 except (FileNotFoundError,ProcessLookupError,PermissionError):return False

def main():
 write(ROOT/'state/failed_worker_guard.json',dict(status='MONITORING_FAILED_JOBS_ONLY',pid=os.getpid()))
 signaled={};seen={}
 while True:
  file=ROOT/'state/dispatcher.json';state=json.loads(file.read_text()) if file.exists() else {}
  for name,pid in state.get('running',{}).items():
   status=ROOT/'raw'/name/'status.json'
   if not status.exists():continue
   data=json.loads(status.read_text())
   if data.get('status')!='FAILED':continue
   key=(name,pid);seen.setdefault(key,time.monotonic())
   if time.monotonic()-seen[key]<15 or not verified(pid,name):continue
   if key not in signaled:
    append(ROOT/'archives/failed_worker_guard_actions.jsonl',dict(name=name,pid=pid,time=time.time(),
     original_status=data,signal='SIGTERM',scope='Verified owned FAILED job session. Main controller and other sessions are not targets.'))
    os.killpg(pid,signal.SIGTERM);signaled[key]=time.monotonic()
   elif time.monotonic()-signaled[key]>10 and verified(pid,name):
    append(ROOT/'archives/failed_worker_guard_actions.jsonl',dict(name=name,pid=pid,time=time.time(),signal='SIGKILL',scope='Same verified FAILED session did not stop after SIGTERM'))
    os.killpg(pid,signal.SIGKILL)
  mainfile=ROOT/'state/main_campaign.json';campaign=json.loads(mainfile.read_text()) if mainfile.exists() else {}
  if campaign.get('status')=='COMPLETE_REVIEW_REQUIRED':break
  controller=campaign.get('pid')
  if controller and not Path(f'/proc/{controller}').exists() and not state.get('running'):break
  time.sleep(2)
 write(ROOT/'state/failed_worker_guard.json',dict(status='FINISHED',pid=os.getpid(),terminated_failed_sessions=len(signaled)))

if __name__=='__main__':main()
