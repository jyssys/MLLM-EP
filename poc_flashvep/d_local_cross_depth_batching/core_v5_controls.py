"""Exclude a diagnostic-only result concatenation from separate execution."""
import time
from .prepare import *
from .run import execute


def main():
    initialize()
    write(ROOT/'state/core_control_v5.json',dict(time=time.time(),
        reason='Earlier separate primitive timing included an output concatenation used only by the equality test. Exclude it from timing.',
        scope='State parity from earlier cells remains evidence. V5 is primary for full-core separate-versus-batched timing. Dense matched-label controls were already free of this concatenation.'))
    for model in ('ouro','huginn'):
        cfg=configuration(model)
        cfg.update(name=f'micro_{model}_strict_grid_v5',worker='operator_micro',micro_strict_geometry=True,
            micro_cases=24,micro_grid=[dict(chunk=c,batch=b,aligned=a) for c in (16,32,64,128) for b in (1,2,4) for a in (True,False)])
        save(cfg);execute(cfg['name'])
        cfg=configuration(model)
        cfg.update(name=f'micro_{model}_kernel_profile_v5',worker='operator_micro',micro_strict_geometry=True,
            micro_cases=1,micro_grid=[dict(chunk=32,batch=2,aligned=False)],micro_profile=True)
        save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
