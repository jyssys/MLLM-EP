"""Resume a serial clean lane and generate the next declared search stage."""
import argparse
import subprocess
import importlib
from .common import *


def terminal(name):
    skip=ROOT/'state/skipped_jobs.json'
    if skip.exists() and name in json.loads(skip.read_text()):return 'SKIPPED'
    folder=ROOT/'raw'/name;status=folder/'status.json';result=folder/'dispatcher_result.json'
    if result.exists():
        code=json.loads(result.read_text())['returncode']
        if code:return 'FAILED'
        if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':return 'COMPLETE'
    return 'PENDING' if not folder.exists() else 'INCOMPLETE'


def main():
    p=argparse.ArgumentParser();p.add_argument('--max-hours',type=float,default=24.);p.add_argument('--wait-pid',type=int,action='append',default=[]);a=p.parse_args();setup()
    while any((Path('/proc')/str(pid)).exists() for pid in a.wait_pid):time.sleep(2)
    plan=ROOT/'configs/study_plan.json';controller=ROOT/'state/controller.json'
    if not plan.exists():write(plan,json.loads((ROOT/'configs/initial_gate_plan.json').read_text()))
    while time.time()<START_EPOCH+a.max_hours*3600:
        names=json.loads(plan.read_text())['configs'];pending=[]
        for filename in names:
            cfg=json.loads((ROOT/'configs'/filename).read_text());state=terminal(cfg['name'])
            if state=='PENDING':
                from .live_audit import blocked,PERFORMANCE_PHASES
                if cfg.get('phase') in PERFORMANCE_PHASES and blocked(cfg['model'],cfg):
                    p=ROOT/'state/skipped_jobs.json';skip=json.loads(p.read_text()) if p.exists() else {}
                    skip[cfg['name']]=dict(time=time.time(),status='NOT_RUN_BLOCKED_CORRECTNESS',model=cfg['model']);write(p,skip)
                else:pending.append(filename)
            elif state=='INCOMPLETE':raise RuntimeError('Do not overwrite an incomplete namespace: '+cfg['name'])
        if pending:
            from . import parallel_qualification
            importlib.reload(parallel_qualification)
            run_qualification=parallel_qualification.run;PHASES=parallel_qualification.PHASES
            first=json.loads((ROOT/'configs'/pending[0]).read_text())
            if first.get('phase') in PHASES:
                batch=[name for name in pending if json.loads((ROOT/'configs'/name).read_text()).get('phase') in PHASES]
                write(controller,dict(time=time.time(),status='PARALLEL_QUALIFICATION',jobs=batch))
                run_qualification(batch)
                continue
            name=pending[0];write(controller,dict(time=time.time(),status='RUNNING',next=name,pending_count=len(pending)))
            print('NEXT',name,flush=True)
            subprocess.run([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.dispatch',name,'--keep-quiet'],cwd=REPO)
            from . import live_audit
            importlib.reload(live_audit)
            live_audit.verify(json.loads((ROOT/'configs'/name).read_text()))
            continue
        if (ROOT/'state/ENABLE_SEARCH').exists():
            from . import search
            importlib.reload(search)
            if search.advance():continue
            write(controller,dict(time=time.time(),status='SEARCH_TERMINAL_REQUIRES_FINAL_REVIEW'));break
        write(controller,dict(time=time.time(),status='WAITING_FOR_CPU_SEARCH_PREPARATION'))
        from .resources import inherited as resources
        if not resources.processes():resources.restore(list(range(8)))
        time.sleep(15)
    else:write(controller,dict(time=time.time(),status='STOPPED_OPERATIONAL_24H_CHECKPOINT'))
    from .resources import inherited as resources
    if not resources.processes():resources.restore(list(range(8)))


if __name__=='__main__':main()
