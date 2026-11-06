"""Run bounded Native controls before selecting an observer or adaptive method."""
import time
from .prepare import *
from .run import execute


def main():
    prepare()
    for model in ('ouro','huginn'):
        for observer in ('off','host','sparse','graph'):
            name=f'a_{model}_heavy_{observer}_s1_n128'
            cfg=configuration(model,name,observer=observer)
            cfg.update(idle_observer_roles=['prefill','decode'],idle_trace_seconds=8.,idle_event_stride=16)
            save(cfg)
            write(ROOT/'state/phase_a.json',dict(time=time.time(),status='RUNNING',name=name))
            execute(name)
    write(ROOT/'state/phase_a.json',dict(time=time.time(),status='INITIAL_CONTROLS_COMPLETE'))


if __name__=='__main__':main()
