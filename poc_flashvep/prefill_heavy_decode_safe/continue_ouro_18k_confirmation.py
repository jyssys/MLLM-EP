"""Require the unchanged pilot gates before independent Native confirmation."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .envelope_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/graph_before_model_18k_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    selected=Path(json.loads((ROOT/'state/current_ouro_observer_choice.json').read_text())['plan'])
    path=ROOT/'analysis'/(selected.stem+'_results.json');decision=dict(time=time.time(),run=False)
    if not path.exists():decision['reason']='No valid completed graph-envelope pilot'
    else:
        value=json.loads(path.read_text());pilot=value['trace']['prefill'];name=value['metrics']
        capture=json.loads(selected.read_text())['jobs'][0]
        host=screen(capture,window=(pilot['left'],pilot['right']))
        eligible=(value['qualified'] and pilot['occupied_union_percent']>=80
            and host['queue']['half_growth']>0 and host['queue']['maximum']>=2
            and host['P_service_fraction']>=.85 and host['metrics']['Decode_SLO_screen'])
        decision.update(run=eligible,observer_qualified=value['qualified'],host=host,
            reason='Native pilot passes the unchanged measurement, pressure, service, and Decode gates' if eligible else 'At least one Native pilot gate failed')
    write(ROOT/'state/ouro_18k_confirmation_entry.json',decision)
    if decision['run']:
        plan=json.loads((ROOT/'state/ouro_18k_confirmation_v4_plan.json').read_text())
        if sequence([plan['jobs'][0]],'ouro18k_native_N32'):
            for i,p in enumerate(plan['plans']):
                cfg=json.loads(Path(p).read_text())
                if not sequence([cfg['clean_reference'],*cfg['jobs']],f'ouro18k_confirm_{i}'):break
                value=analyze(p);t=value['trace']['prefill']
                screen(cfg['jobs'][0],window=(t['left'],t['right']))
    write(ROOT/'state/ouro_18k_confirmation_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
