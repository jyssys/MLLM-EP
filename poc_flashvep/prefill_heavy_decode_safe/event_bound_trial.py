"""Use validated GPU enclosures only if graph correlation remains unresolved."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .device_bound_analysis import analyze
from .host_screen import screen


def prepare():
    old=json.loads((ROOT/'state/graph_causal_scope_18k_plan.json').read_text())
    common=dict(discovery_D_device_bounds=True,activity_D_graph_enclosures=False,
        activity_causal_record_scope=False,activity_measurement_start_device_fence=False,
        discovery_D_execution_events=False,discovery_natural_readiness=True,
        observed_source_cuts=[3],activity_before_model=True)
    parent=old['state8'];cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
    state='event_bound_ouro_L18432_observer_state8';cfg.update(common)
    cfg.update(name=state,activity_D_disabled=True,required_qualification=parent,
        phase='MULTI_STREAM_DEVICE_EVENT_BOUND_EXACT_STATE')
    save(cfg);jobs=[state];names={}
    for mode in ('clean','coverage','capture','repeat'):
        parent=old['clean_reference'];cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
        cfg.update(common);name=parent.replace('graph_causal_scope_','event_bound_').rsplit('_clean_w24',1)[0]+'_'+mode+'_w24'
        cfg.update(name=name,discovery_activity=mode in ('coverage','capture'),
            discovery_event_observer=mode in ('coverage','capture'),
            discovery_D_device_bounds=mode in ('coverage','capture'),activity_D_disabled=mode=='capture',
            required_qualification=state if mode in ('coverage','capture') else 'extended_ouro_L18432_native_reference_state8',
            phase='VALIDATED_DEVICE_BOUND_MATCHED_NATIVE_CONTROL')
        save(cfg);jobs.append(name);names[mode]=name
    path=ROOT/'state/event_bound_18k_plan.json'
    write(path,dict(state8=state,all_jobs=jobs,clean_reference=names['clean'],coverage_run=names['coverage'],
        capture=names['capture'],repeat=names['repeat'],padding_ms=2.,
        scope='First verify every recorded D kernel lies inside padded GPU-event enclosures in a separate full-node diagnostic. Then disable D CUPTI and retain P kernel tracing. The complement is only a conservative inactivity bound. Exact D kernel busy/idle and its complete distribution remain NOT_MEASURED. The five-percent whole-run, local-tail, and Native A/A screens remain unchanged.'))
    return path,jobs


def main():
    initialize();path,jobs=prepare()
    while not (ROOT/'state/graph_runtime_scope_18k_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    previous=json.loads((ROOT/'state/graph_runtime_scope_18k_entry.json').read_text())
    entry=dict(run=False)
    if previous.get('run'):
        entry.update(reason='The earlier observer qualified. No extra collector trial is necessary.',selected='runtime_scope')
    elif sequence(jobs,'event_bound_18k_controller'):
        value=analyze(path);t=value['trace']['prefill'];cfg=json.loads(path.read_text())
        host=screen(cfg['capture'],window=(t['left'],t['right']))
        entry.update(observer_qualified=value['qualified'],host=host,
            run=value['qualified'] and t['occupied_union_percent']>=80 and host['queue']['half_growth']>0
                and host['queue']['maximum']>=2 and host['P_service_fraction']>=.85 and host['metrics']['Decode_SLO_screen'])
    write(ROOT/'state/event_bound_18k_entry.json',entry)
    write(ROOT/'state/event_bound_18k_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
