"""Qualify mixed ownership states. Forced cuts are not an adaptive policy."""
import copy
from .prepare import *
from .run import execute


def configuration_for(model,count,mixed):
    method='loop3' if model=='ouro' else 'loop31'
    source=DEEP/'configs'/f'b_{model}_{method if mixed else "native"}_reserved8_state8.json'
    cfg=copy.deepcopy(json.loads(source.read_text()))
    name=f'q_{model}_{"forced_mixed" if mixed else "native"}_state{count}'
    cfg.update(name=name,count=count,state_probe=True,verify_transfer=True,observe_service_timing=False,
        idle_campaign=True,idle_observer='off',dlocal_campaign=False,D_local_probe_output_pacing=False,
        phase='FORCED_MIXED_CUT_PROTOCOL_QUALIFICATION_NOT_ONLINE_ADAPTATION',
        calibration_serial=False,observe_device_execution=False,arrival_offsets=[0.]*count,
        arrival_seed=None,arrival_rate=None,arrival_pattern='SIMULTANEOUS_SUBMISSION_QUALIFICATION',
        deep_complete_D_geometry_warmup=False,timeout=900,startup_timeout=300)
    cfg['dataset']=f'heldout_{model}_L{cfg["length"]}.json'
    pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    cfg['prompt_ids']=[v['prompt_id'] for v in pool[:count]];cfg['request_caps']=[cfg['cap']]*count
    if mixed:
        cuts=(1,2,3,4) if model=='ouro' else (24,28,30,31,32)
        cfg['native_request_cuts']=[cuts[i%len(cuts)] for i in range(count)]
        cfg['forced_cut_scope']='Protocol coverage only. No Safe Slack estimator or online decision is exercised.'
    return cfg


def main():
    prepare()
    from ..native_loop_2d_oracle import qualification
    qualification.initialize=initialize;qualification.execute=execute
    results=[]
    for model in ('ouro','huginn'):
        for count in (8,32):
            native=configuration_for(model,count,False);save(native)
            reference=qualification.qualify(native['name']);results.append(reference)
            write(ROOT/'state/forced_mixed_qualification.json',dict(status='RUNNING',results=results))
            if not reference['passed']:break
            cfg=configuration_for(model,count,True);save(cfg)
            result=qualification.qualify(cfg['name'],native['name']);results.append(result)
            write(ROOT/'state/forced_mixed_qualification.json',dict(status='RUNNING',results=results))
            if not result['passed']:break
    write(ROOT/'state/forced_mixed_qualification.json',dict(status='COMPLETE',results=results,
        scope='Forced heterogeneous cuts qualify connector and state lifetimes only. They do not prove Safe Slack admission, Decode SLO, or adaptive performance.'))


if __name__=='__main__':main()
