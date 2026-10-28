"""Restore only idle owned GPU leases after a launched discovery job exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT,REPO,setup,write
from . import resources

def live(pid):
 try:
  proc=Path(f'/proc/{pid}')
  return proc.stat().st_uid==os.getuid() and (proc/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False

def run():
 setup();expected={'nixl_pair01','ouro_native_smoke','huginn_native_smoke','ouro_pd_smoke'}
 while True:
  path=ROOT/'commands.jsonl'
  rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
  jobs={r['name']:r for r in rows if r['name'] in expected}
  active=[r for r in jobs.values() if live(r['pid'])]
  reserved={g for r in active for g in r['gpus']}
  for r in jobs.values():
   if not live(r['pid']) and not set(r['gpus'])&reserved:resources.base.restore(r['gpus'])
  write(ROOT/'state/initial_jobs.json',dict(status='RUNNING' if active or len(jobs)<4 else 'COMPLETE_REVIEW_REQUIRED',
   active={r['name']:r['pid'] for r in active},time=time.time()))
  if len(jobs)==4 and not active:break
  time.sleep(3)

def main():
 setup();p=argparse.ArgumentParser();p.add_argument('action',choices=('start','run'));a=p.parse_args()
 if a.action=='run':run()
 else:
  with (ROOT/'logs/lease_watch.log').open('a') as f:
   child=subprocess.Popen([sys.executable,'-m','poc_flashvep.looped_pd_abc.lease_watch','run'],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  print('LEASE_WATCH',child.pid)

if __name__=='__main__':main()
