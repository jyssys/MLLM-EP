"""Attempt a bounded scope diagnosis before deciding the measurement verdict."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .envelope_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not all((ROOT/'state'/p).exists() for p in ('native_three_slots_complete.json','ouro_18k_gap_cost_complete.json')):
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    path=ROOT/'state/graph_causal_scope_18k_plan.json';cfg=json.loads(path.read_text());entry=dict(run=False)
    if sequence(cfg['all_jobs'],'graph_causal_scope_18k_controller'):
        value=analyze(path);t=value['trace']['prefill'];host=screen(cfg['jobs'][0],window=(t['left'],t['right']))
        eligible=(value['qualified'] and t['occupied_union_percent']>=80 and host['queue']['half_growth']>0
            and host['queue']['maximum']>=2 and host['P_service_fraction']>=.85 and host['metrics']['Decode_SLO_screen'])
        entry.update(run=eligible,observer_qualified=value['qualified'],host=host)
        if eligible:
            plan=json.loads((ROOT/'state/ouro_causal_confirmation_plan.json').read_text())
            if sequence([plan['jobs'][0]],'ouro_causal_native_state32'):
                for i,p in enumerate(plan['plans']):
                    c=json.loads(Path(p).read_text())
                    if not sequence([c['clean_reference'],*c['jobs']],f'ouro_causal_confirmation_{i}'):break
                    v=analyze(p);t=v['trace']['prefill'];screen(c['jobs'][0],window=(t['left'],t['right']))
    write(ROOT/'state/ouro_causal_confirmation_entry.json',entry)
    write(ROOT/'state/ouro_causal_confirmation_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
