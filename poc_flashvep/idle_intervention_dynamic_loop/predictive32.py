"""A future admission forecast must not grant immediate execution."""
from .policy import *


def prototype(model,arm,safety='relative',count=32):
    cfg=configured(model,f'pred_{model}_{arm}_{safety}_clean{count}',arm,count)
    cfg.update(predictive_admission=True,admission_horizon_ms=500.,maximum_quantum_ms=200.,
        guard_ms=0.,policy_revision='predictive-v2',
        phase='PREDICTIVE_ADMISSION_SELECTION_NOT_HELD_OUT_CONFIRMATION',
        margin_scope='The candidate cost already includes its frozen p99 guard. Decode prediction retains its own guard. No duplicate execution guard is added. Admission forecasts future opportunities. Every actual execution still uses the current deadline budget.')
    if safety=='absolute':
        frozen=metrics(f'cap_{model}_native_calibration128')['frozen_SLO']
        cfg['native_ITL_target_ms']=min(frozen['ITL_ms'],.9*frozen['TPOT_ms'])
        cfg['feedback_ITL_limit_ms']=cfg['native_ITL_target_ms']
        cfg['safety_scope']='Frozen absolute Decode SLO selection. Native-relative 5% is evaluated separately. This is not a non-inferiority success by definition.'
    if cfg.get('execution_candidates'):
        terminal=[c['cost_ms'] for c in cfg['execution_candidates'] if c.get('terminal') and c['operation']=='recurrent']
        cfg['minimum_quantum_ms']=min(terminal)
    return cfg


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for safety in ('relative','absolute'):
            for arm in ('loop','tile'):
                cfg=prototype(model,arm,safety);save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
