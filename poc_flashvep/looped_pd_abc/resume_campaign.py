"""Resume controller work at a completed-run boundary after a user scope change."""
import argparse
import os
import subprocess
import time
import traceback
from .common import ROOT,REPO,PYTHON,write
from .scope import active_models

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--from-phase',choices=('C','scale','profiles'),default='C')
 args=parser.parse_args()
 from .run_benchmarks import wait_for_quiet_lease,apply_identity_gates,profiles
 from .dispatch import execute
 from .analyze import summarize
 from . import resources
 def state(status,phase,**extra):
  write(ROOT/'state/main_campaign.json',dict(status=status,phase=phase,pid=os.getpid(),
   active_models=active_models(),time=time.time(),**extra))
 try:
  state('RUNNING','resume_wait_for_existing_worker')
  wait_for_quiet_lease()
  phases=('C','scale') if args.from_phase=='C' else ('scale',) if args.from_phase=='scale' else ()
  for phase in phases:
   state('RUNNING',phase)
   subprocess.run([PYTHON,'-m','poc_flashvep.looped_pd_abc.campaign',phase],cwd=REPO,check=True)
   manifest=ROOT/'configs'/f'{phase}_campaign.json';apply_identity_gates(manifest)
   execute(manifest);summarize()
  state('RUNNING','profiles');profiles();summarize()
  state('COMPLETE_REVIEW_REQUIRED','review',pending=['inspect failures and CIs','conditional confirmation','final reports','scoped commit and push'])
 except BaseException as error:
  state('FAILED_REVIEW_REQUIRED','controller_failure',error=repr(error),traceback=traceback.format_exc())
  resources.base.restore(list(range(8)))
  raise

if __name__=='__main__':main()
