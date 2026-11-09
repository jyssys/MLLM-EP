"""Own a bounded queue. Restore verified missing burners between experiments."""
import argparse
import fcntl
import traceback
from .common import *
from .run import execute


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.common import resources
    from ..adaptive_idle_loop_continuation.monitor import restore_missing_burners
    from ..adaptive_idle_loop_continuation import deadline_guard
    import subprocess
    deadline_guard.configure=configure;deadline_guard.ROOT=ROOT;deadline_guard.GPU_DEADLINE=GPU_DEADLINE
    guard=subprocess.Popen(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m',
        'poc_flashvep.idle_intervention_dynamic_loop.owner','--guard'],start_new_session=True)
    write(ROOT/'state/queue_owner.json',dict(pid=os.getpid(),guard=guard.pid,time=time.time(),deadline=GPU_DEADLINE))
    while time.time()<LONG_START_CUTOFF and not (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
        plan=ROOT/'state/queue.json';queue=json.loads(plan.read_text()) if plan.exists() else []
        next_job=next((name for name in queue if not (ROOT/'raw'/name/'dispatcher_result.json').exists()),None)
        if next_job:
            config=json.loads((ROOT/'configs'/(next_job+'.json')).read_text())
            if COMPLETION_WINDOW:
                fits=(next_job.startswith(tuple(COMPLETION_WINDOW['allowed_prefixes']))
                    and not config.get('state_probe')
                    and time.time()+config.get('timeout',300)+config.get('startup_timeout',240)+60<GPU_DEADLINE)
                if not fits:
                    write(ROOT/'raw'/next_job/'dispatcher_result.json',dict(complete=False,
                        status='NOT_RUN_BOUNDED_COMPLETION_WINDOW',elapsed_seconds=0.,time=time.time(),
                        scope='The declared startup timeout, run timeout, and cleanup margin do not fit the unchanged GPU deadline. No model execution started.'))
                    continue
            repeat=config.get('repeat_safety_references',[])
            if repeat:
                from .analyze import metrics
                from .qualification import qualify
                unsafe=[]
                for previous in repeat:
                    pconfig=json.loads((ROOT/'configs'/(previous+'.json')).read_text())
                    result=metrics(previous)
                    if not result.get('complete') or not result.get('Decode_SLO_screen'):
                        unsafe.append(dict(name=previous,reason='INCOMPLETE_OR_ABSOLUTE_DECODE_GATE_FAILURE'))
                    elif not qualify(previous,pconfig.get('comparison_reference')).get('passed'):
                        unsafe.append(dict(name=previous,reason='CORRECTNESS_OR_LIFETIME_FAILURE'))
                if unsafe:
                    write(ROOT/'raw'/next_job/'dispatcher_result.json',dict(complete=False,
                        status='NOT_RUN_PRIOR_SAFETY_GATE',elapsed_seconds=0.,time=time.time(),prior_failures=unsafe))
                    continue
            gates=ROOT/'state/resumption_gate_dependencies.json'
            dependencies=json.loads(gates.read_text()).get(next_job,[]) if RESUMPTION and gates.exists() else []
            if dependencies:
                from .qualification import qualify
                results={}
                for name in dependencies:
                    config=json.loads((ROOT/'configs'/(name+'.json')).read_text());model=config['model']
                    ref=f'a_{model}_native_state8' if config['count']==8 else f'n32_{model}_native_full_state32'
                    results[name]=qualify(name,ref)
                if not all(value.get('passed') for value in results.values()):
                    write(ROOT/'raw'/next_job/'dispatcher_result.json',dict(complete=False,
                        status='NOT_RUN_QUALIFICATION_GATE',elapsed_seconds=0.,time=time.time(),
                        dependencies={name:value.get('passed',False) for name,value in results.items()}))
                    continue
            try:execute(next_job)
            except BaseException:
                write(ROOT/'state'/('owner_error_'+next_job+'.json'),dict(time=time.time(),traceback=traceback.format_exc()))
                return
        else:
            with (ROOT/'state/quiet_lane.lock').open('a+') as lane:
                try:fcntl.flock(lane,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:restore_missing_burners(resources())
            time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--guard',action='store_true');args=parser.parse_args()
    if args.guard:
        configure()
        from ..adaptive_idle_loop_continuation import deadline_guard
        deadline_guard.configure=configure;deadline_guard.ROOT=ROOT;deadline_guard.GPU_DEADLINE=GPU_DEADLINE
        deadline_guard.main()
    else:main()
