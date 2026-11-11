"""Require independent controlled starts after a qualified enclosure pilot."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .device_bound_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/event_bound_18k_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    entry=json.loads((ROOT/'state/event_bound_18k_entry.json').read_text())
    if not entry.get('run'):
        write(ROOT/'state/event_bound_confirmation_complete.json',dict(complete=True,run=False,reason='Pilot did not pass all entry gates',time=time.time()))
        return
    old=json.loads((ROOT/'state/ouro_causal_confirmation_plan.json').read_text())
    pilot=json.loads((ROOT/'state/event_bound_18k_plan.json').read_text());plans=[];jobs=[]
    for p in old['plans']:
        source=json.loads(Path(p).read_text());names=[]
        for parent in [source['clean_reference'],*source['jobs']]:
            cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('confirm_scope_','confirm_event_bound_')
            active=bool(cfg['discovery_activity'])
            cfg.update(name=name,discovery_D_device_bounds=active,activity_D_disabled=active,
                activity_D_graph_enclosures=False,activity_causal_record_scope=False,
                activity_measurement_start_device_fence=False,discovery_D_execution_events=False,
                discovery_natural_readiness=True,observed_source_cuts=[3],
                required_qualification=pilot['state8'] if active else old['jobs'][0],
                phase='INDEPENDENT_NATIVE_EVENT_INACTIVITY_BOUND_CONFIRMATION')
            save(cfg);names.append(name);jobs.append(name)
        path=Path(p).with_name(Path(p).name.replace('confirm_scope_','confirm_event_bound_'))
        write(path,dict(clean_reference=names[0],capture=names[1],repeat=names[2],
            coverage_run=pilot['coverage_run'],padding_ms=2.,
            scope='Coverage comes from a separate same-runtime diagnostic. Exact D kernel idle remains NOT_MEASURED. All latency screens remain unchanged.'))
        plans.append(str(path))
    write(ROOT/'state/event_bound_confirmation_plan.json',dict(state32=old['jobs'][0],jobs=jobs,plans=plans))
    if sequence([old['jobs'][0]],'event_bound_native_N32'):
        for i,path in enumerate(plans):
            cfg=json.loads(Path(path).read_text())
            if not sequence([cfg['clean_reference'],cfg['capture'],cfg['repeat']],f'event_bound_N128_confirmation_{i}'):break
            value=analyze(path);t=value['trace']['prefill'];screen(cfg['capture'],window=(t['left'],t['right']))
    write(ROOT/'state/event_bound_confirmation_complete.json',dict(complete=True,run=True,time=time.time()))


if __name__=='__main__':main()
