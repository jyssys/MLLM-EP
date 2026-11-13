"""Test a predefined static fraction as an explicitly exploratory control."""
import fcntl
from .common import *
from .run import execute
from .analyze import metrics,compare


def main():
    initialize()
    write(ROOT/'state/exploratory_fraction50_design.json',dict(time=time.time(),
        trigger='The first held-out COST start migrated60/128 requests. The frozen discovery-rate static control migrated about31%.',
        hypothesis='Check whether a fixed50% mask explains the apparent state-aware benefit without changing the primary selection.',
        fraction=.5,seeds=[2026111341,2026111342,2026111343,2026111344,2026111345],
        scope='Exploratory control added after viewing the first held-out start. Not a pre-registered primary arm. Static50% was a predefined validation candidate. No SLO, predictor, or primary selection changes.'))
    deadline=time.monotonic()+18000
    while not (ROOT/'analysis/heldout_results.json').exists():
        assert time.monotonic()<deadline,'Primary confirmation did not finish'
        time.sleep(5)
    selection=json.loads((ROOT/'configs/heldout_selection.json').read_text());rows=[]
    try:
        for seed in selection['primary_Output16_arrival_seeds']:
            base_name=f'heldout_O16_N128_s{seed}_MatchedFraction_v1'
            cfg=json.loads((ROOT/'configs'/(base_name+'.json')).read_text())
            cfg.update(name=f'heldout_O16_N128_s{seed}_FixedFraction50_exploratory_v1',
                migration_fraction=.5,evaluation_arm='FixedFraction50',
                phase='EXPLORATORY_FIXED50_RATE_CONTROL_AFTER_FIRST_HELDOUT_RESULT',
                fraction_scope='Fixed50% request-ID blocks. Explicitly exploratory. No held-out tuning of the primary policy.')
            if not execute(save(cfg),restore=False):return
            gate=compare('heldout_O16_N128_s2026111341_Native_v1',cfg['name'])
            row=dict(name=cfg['name'],arm=cfg['evaluation_arm'],cap=16,arrival_seed=seed,
                output_gate=gate,metrics=metrics(cfg['name']),exploratory=True)
            rows.append(row);write(ROOT/'state/exploratory_fraction50_progress.json',dict(results=rows))
            print('EXPLORATORY_FIXED50',cfg['name'],gate['passed'],row['metrics']['TTFT_p95_ms'],flush=True)
            if not gate['passed']:return
        write(ROOT/'analysis/exploratory_fraction50_results.json',dict(results=rows,exploratory=True))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
