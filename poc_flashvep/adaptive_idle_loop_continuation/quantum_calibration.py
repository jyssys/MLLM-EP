"""Qualify and time equal-quantum primitives. Do not call them adaptive serving."""
import copy
from .prepare import *
from .run import execute
from .primitive_full import primitive


def main():
    prepare()
    from ..native_loop_2d_oracle import qualification
    qualification.initialize=initialize;qualification.execute=execute
    results=[]
    for model in ('ouro','huginn'):
        reference=None
        token='token3072' if model=='ouro' else 'token1984'
        loop='loop3' if model=='ouro' else 'loop31'
        for method in ('native',loop,token):
            # Use the already frozen deep configuration for the state test.
            source=DEEP/'configs'/f'b_{model}_{method}_reserved8_state8.json'
            cfg=copy.deepcopy(json.loads(source.read_text()))
            cfg.update(name=f'q_{model}_{method}_Dquantum1_state8',count=8,
                idle_campaign=True,idle_observer='off',idle_D_loop_quantum=1,dlocal_campaign=False,
                D_local_probe_output_pacing=False,observe_service_timing=False,deep_complete_D_geometry_warmup=False,
                arrival_rate=None,arrival_seed=None,arrival_offsets=[0.]*8,
                phase='EQUAL_D_QUANTUM_EXACTNESS_NOT_ADAPTIVE_SERVING')
            cfg['dataset']=f'heldout_{model}_L{cfg["length"]}.json'
            pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:8]
            cfg['prompt_ids']=[v['prompt_id'] for v in pool];cfg['request_caps']=[cfg['cap']]*8
            save(cfg)
            result=qualification.qualify(cfg['name'],reference);results.append(result)
            write(ROOT/'state/quantum_calibration.json',dict(status='RUNNING',results=results))
            if not result['passed']:
                if method=='native':break
                continue
            if method=='native':reference=cfg['name'];continue
            assert reference
            timed=primitive(model,'Dquantum1_'+method,cfg.get('native_partial_cut'))
            if cfg.get('native_frontier') is not None:timed['native_frontier']=cfg['native_frontier']
            timed.update(idle_D_loop_quantum=1,phase='EQUAL_D_QUANTUM_SERIAL_COST_NOT_CLEAN_SERVING')
            save(timed);results.append(dict(name=timed['name'],complete=execute(timed['name']),
                state_qualification=cfg['name']))
            write(ROOT/'state/quantum_calibration.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/quantum_calibration.json',dict(status='COMPLETE',results=results))


if __name__=='__main__':main()
