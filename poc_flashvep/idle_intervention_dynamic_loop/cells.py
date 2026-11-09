"""Create explicit calibration cells. These fixed cuts are not adaptive results."""
from .prepare import *


def intervention(model,width,count=8,state=True,dynamic=False):
    name=f'b2_{model}_{"dynamic" if dynamic else "static_last"}_tile{width}_state{count}'
    cfg=configuration(model,name,count,state=state,method='dynamic_loop' if dynamic else 'static_intervention')
    depth=cfg['depth']
    cfg.update(D_tile_tokens=width,D_loop_quantum=1,minimum_quantum_ms=0.,
        empty_decode_budget_ms=200.,maximum_quantum_ms=0.,
        native_ITL_target_ms=0.,decode_reserve_ms=0.,guard_ms=1.,
        qualification_allow_unpressured=True,remaining_cost_ms={str(i):50.*i for i in range(1,depth)},
        admission_limit_ms=1000.,handoff_upper_ms=100.,minimum_P_pressure=1,
        phase='STATE_QUALIFICATION_NOT_DECODE_SAFETY_EVIDENCE')
    if not dynamic:cfg['native_partial_cut']=depth-1
    # A zero threshold permits protocol coverage during heavy state checks.
    # These cells do not establish Decode-safe admission or timing.
    return cfg


def main():
    prepare();names=[]
    for model in ('ouro','huginn'):
        for dynamic in (False,True):
            cfg=intervention(model,4096 if model=='ouro' else 2048,dynamic=dynamic)
            save(cfg);names.append(cfg['name'])
    queue=json.loads((ROOT/'state/queue.json').read_text())
    write(ROOT/'state/queue.json',queue+names)


if __name__=='__main__':main()
