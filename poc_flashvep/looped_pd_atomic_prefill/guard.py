"""Observe durable progress. Resume only a stopped, fully terminal queue."""
import subprocess
from .common import *


def processes(suffix):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            words=(p/'cmdline').read_bytes().split(b'\0')
            module=('poc_flashvep.looped_pd_atomic_prefill.'+suffix).encode()
            if p.stat().st_uid==os.getuid() and b'-m' in words and module in words:found.append(int(p.name))
        except (FileNotFoundError,PermissionError):pass
    return found


def main():
    retries=0
    while not (ROOT/'state/CLOSED.json').exists():
        stage_path=ROOT/'state/search_stage.json';stage=json.loads(stage_path.read_text()) if stage_path.exists() else {}
        controllers=processes('campaign');dispatchers=processes('dispatch')
        complete=failed=0;incomplete=[]
        for folder in (ROOT/'raw').iterdir():
            if not folder.is_dir():continue
            p=folder/'dispatcher_result.json'
            if p.exists():
                if json.loads(p.read_text())['returncode']:failed+=1
                else:complete+=1
            elif not dispatchers:incomplete.append(folder.name)
        state=dict(time=time.time(),start_epoch=START_EPOCH,elapsed_hours=(time.time()-START_EPOCH)/3600,
            operational_checkpoint_epoch=START_EPOCH+86400,stage=stage.get('stage'),controllers=controllers,
            dispatchers=dispatchers,completed_server_starts=complete,failed_server_starts=failed,
            incomplete_namespaces=incomplete,automatic_controller_restarts=retries)
        write(ROOT/'budget.json',state)
        terminal=stage.get('stage') in ('GPU_SEARCH_COMPLETE_ANALYSIS_PENDING','FINAL_COMPLETE','STOPPED_USER_11H_DEADLINE') or time.time()>=START_EPOCH+86400
        if not terminal and not controllers and not dispatchers and not incomplete and not processes('extension_manager') and retries<3:
            with (ROOT/'logs/campaign_guard_resume.log').open('a') as log:
                child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.campaign'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            retries+=1;append(ROOT/'state/controller_reloads.jsonl',dict(time=time.time(),new_controller=child.pid,
                reason='Owned controller stopped. Every previous namespace is terminal. Resume skips complete and failed runs; it does not retry their namespaces.'))
        time.sleep(10)


if __name__=='__main__':main()
