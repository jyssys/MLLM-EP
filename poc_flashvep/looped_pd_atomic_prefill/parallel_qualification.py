"""Parallelize correctness only. Clean service jobs retain an exclusive lease."""
import subprocess
from .common import *

PHASES={'correctness','long_correctness','chunk_correctness','final_correctness'}


def run(filenames,maximum=3):
    waiting=[(name,json.loads((ROOT/'configs'/name).read_text())) for name in filenames]
    from .resources import inherited as resources
    needed={gpu for name,cfg in waiting for gpu in cfg['gpus']}
    # Per-GPU supervisors can be stopped without affecting another idle GPU.
    for gpu in sorted(set(range(8))-needed):resources.restore([gpu])
    active={};restore_time=0.
    while waiting or active:
        for pid,(child,name,cfg) in list(active.items()):
            if child.poll() is not None:
                append(ROOT/'state/parallel_qualification.jsonl',dict(time=time.time(),event='finished',pid=pid,name=name,returncode=child.returncode,gpus=cfg['gpus']))
                del active[pid]
        occupied={gpu for child,name,cfg in active.values() for gpu in cfg['gpus']}
        for name,cfg in list(waiting):
            if len(active)>=maximum:break
            if occupied.intersection(cfg['gpus']):continue
            child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.dispatch',name,'--keep-quiet'],cwd=REPO)
            active[child.pid]=(child,name,cfg);waiting.remove((name,cfg));occupied.update(cfg['gpus'])
            append(ROOT/'state/parallel_qualification.jsonl',dict(time=time.time(),event='started',pid=child.pid,name=name,gpus=cfg['gpus'],
                scope='Correctness evidence only. Concurrent shared-host activity is not clean latency.'))
        if time.monotonic()-restore_time>10:
            restore_time=time.monotonic()
            for gpu in sorted(set(range(8))-occupied):resources.restore([gpu])
        write(ROOT/'state/parallel_qualification_active.json',dict(time=time.time(),active=[dict(pid=pid,name=name,gpus=cfg['gpus']) for pid,(child,name,cfg) in active.items()],waiting=len(waiting)))
        time.sleep(1)
