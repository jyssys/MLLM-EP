"""Compile and publish only after every managed GPU job is terminal."""
import argparse
import fcntl
import subprocess
from .common import *
from .guard import processes


def run(module,*arguments):
    command=[str(PYTHON),'-m',module,*arguments]
    append(ROOT/'state/closeout_commands.jsonl',dict(time=time.time(),command=command))
    subprocess.run(command,cwd=REPO,check=True)


def terminal_stage():
    extension=ROOT/'configs/extension_budget_20261102.json';terminal=ROOT/'state/extension_terminal_20261102.json'
    if extension.exists() and not terminal.exists():
        if time.time()<json.loads(extension.read_text())['GPU_stop_epoch']:return None
    p=ROOT/'state/search_stage.json';stage=json.loads(p.read_text()) if p.exists() else {}
    control=ROOT/'state/controller.json';controller=json.loads(control.read_text()) if control.exists() else {}
    if stage.get('stage')=='GPU_SEARCH_COMPLETE_ANALYSIS_PENDING':return stage['stage']
    if stage.get('stage')=='STOPPED_USER_11H_DEADLINE':return stage['stage']
    if controller.get('status')=='STOPPED_OPERATIONAL_24H_CHECKPOINT' and not processes('campaign'):
        return 'STOPPED_OPERATIONAL_24H_CHECKPOINT'
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--push',action='store_true');a=parser.parse_args();setup()
    lease=(ROOT/'state/closeout.lock').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=ROOT/'state/closeout_status.json'
    if (ROOT/'state/CLOSED.json').exists():return
    try:
        while True:
            stage=terminal_stage()
            active=processes('dispatch')+processes('campaign')
            if stage and not active:break
            write(marker,dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_GPU_TERMINAL',search_terminal=stage,managed_processes=active))
            if not a.wait:raise RuntimeError('GPU search has not reached a terminal state. No final claim or publication was made.')
            time.sleep(10)
        from .campaign import terminal
        plan=json.loads((ROOT/'configs/study_plan.json').read_text())['configs']
        pending=[];incomplete=[]
        for filename in plan:
            cfg=json.loads((ROOT/'configs'/filename).read_text());state=terminal(cfg['name'])
            if state=='PENDING':pending.append(cfg['name'])
            if state=='INCOMPLETE':incomplete.append(cfg['name'])
        assert not incomplete,('Do not claim completion for an incomplete namespace',incomplete)
        if stage in ('STOPPED_OPERATIONAL_24H_CHECKPOINT','STOPPED_USER_11H_DEADLINE'):
            p=ROOT/'state/skipped_jobs.json';skips=json.loads(p.read_text()) if p.exists() else {}
            for name in pending:skips[name]=dict(time=time.time(),status='NOT_RUN_'+stage)
            write(p,skips);write(ROOT/'state/search_stage.json',dict(time=time.time(),stage=stage,unrun=len(pending)))
        else:assert not pending,('Search stopped with queued jobs',pending)
        write(marker,dict(time=time.time(),pid=os.getpid(),status='CPU_EVIDENCE_REVIEW',stage=stage))
        run('poc_flashvep.looped_pd_atomic_prefill.bootstrap')
        subprocess.run([str(PYTHON),'-m','unittest','poc_flashvep.looped_pd_atomic_prefill.test_contracts','-q'],cwd=REPO,check=True)
        for module in ('analyze','matrix','profiles','native_capacity_controls','classify','uncertainty','sensitivity','figures'):
            run('poc_flashvep.looped_pd_atomic_prefill.'+module,*(['analyze'] if module=='native_capacity_controls' else []))
        run('poc_flashvep.looped_pd_atomic_prefill.reporting')
        from .resources import inherited as resources
        with (ROOT/'state/gpu_lease.lock').open('a+') as gpu_lease:
            fcntl.flock(gpu_lease,fcntl.LOCK_EX)
            assert not processes('dispatch') and not processes('campaign')
            resources.restore(list(range(8)))
            write(ROOT/'state/final_resources.json',dict(time=time.time(),owned_burners=resources.burners(),GPU_processes=resources.processes(),
                scope='Actual ownership-checked restore. Foreign work is not stopped to obtain eight burner processes.'))
        run('poc_flashvep.looped_pd_atomic_prefill.archive')
        run('poc_flashvep.looped_pd_atomic_prefill.publish','--final',*(['--push'] if a.push else []))
        publication=json.loads((ROOT/'state/publish_final.json').read_text())
        value=dict(time=time.time(),pid=os.getpid(),status='FINAL_COMPLETE',GPU_search_terminal=stage,
            publication=publication,report=str(REPO/'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover_final_summary.md'))
        write(marker,value);write(ROOT/'state/CLOSED.json',value)
        write(ROOT/'state/search_stage.json',dict(time=time.time(),stage='FINAL_COMPLETE',publication=publication))
        print(json.dumps(value),flush=True)
    except BaseException as error:
        import traceback
        write(marker,dict(time=time.time(),pid=os.getpid(),status='BLOCKED_CLOSEOUT',error=repr(error),traceback=traceback.format_exc(),
            scope='No successful completion is claimed. Existing GPU jobs and raw evidence remain untouched.'))
        raise


if __name__=='__main__':main()
