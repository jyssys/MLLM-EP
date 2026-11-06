"""Check role-device activation after the first observer controls finish."""
import time
from .prepare import *
from .run import execute


def main():
    prepare()
    while time.time()<LONG_START_CUTOFF:
        path=ROOT/'state/phase_a.json'
        if path.exists() and json.loads(path.read_text())['status']=='INITIAL_CONTROLS_COMPLETE':break
        time.sleep(5)
    for model in ('ouro','huginn'):
        cfg=configuration(model,f'a_{model}_graph_device_context_smoke24',count=24,observer='graph')
        cfg.update(idle_activity_device_context=True,idle_observer_roles=['prefill','decode'],idle_trace_seconds=3.)
        save(cfg);execute(cfg['name'])
    write(ROOT/'state/context_trials.json',dict(time=time.time(),status='COMPLETE'))


if __name__=='__main__':main()
