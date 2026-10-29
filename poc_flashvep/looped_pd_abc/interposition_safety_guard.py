"""Resume only the verified launcher if its interposition process exits unexpectedly."""
import argparse
import os
from pathlib import Path
import signal
import time
from .common import ROOT,write
from .quality_interposition import identity

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--interposer',type=int,required=True)
 parser.add_argument('--controller',type=int,required=True);args=parser.parse_args()
 token=identity(args.controller)
 if not token:raise RuntimeError('No verified owned launcher')
 target=Path('/proc')/str(args.interposer)
 if target.exists() and (target.stat().st_uid!=os.getuid() or b'poc_flashvep.looped_pd_abc.quality_interposition' not in (target/'cmdline').read_bytes()):
  raise RuntimeError('Unexpected interposition process')
 while target.exists():
  try:
   if (target/'stat').read_text().split()[2]=='Z':break
  except FileNotFoundError:break
  time.sleep(2)
 resumed=False
 if identity(args.controller)==token:
  paused=(Path('/proc')/str(args.controller)/'stat').read_text().split()[2] in ('T','t')
  if paused:os.kill(args.controller,signal.SIGCONT);resumed=True
 write(ROOT/'state/interposition_safety_guard.json',dict(status='FINISHED',fallback_resume=resumed,
  controller=args.controller,interposer=args.interposer,time=time.time(),
  scope='Launcher signal only. No GPU worker or unrelated process was signaled.'))

if __name__=='__main__':main()
