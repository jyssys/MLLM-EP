"""Reserve matched controls for a repeated decode-safe Output64 gain."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,resources,save,write
from .analysis import metrics
from .run import execute
from .heldout_baselines import SEEDS
from .matched_policy_controls import control_config,compare_geometry
from .statistics_report import ratio_interval
from .policy_control_statistics import summarize


def completed(name):
    path=ROOT/'raw'/name/'dispatcher_result.json'
    return path.exists() and json.loads(path.read_text()).get('complete')


def main():
    cutoff=json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-4200
    try:
        for qps in (4.8,6.4):
            names={method:[f'heldout_{method}_N128_O64_q{qps:g}_s{seed}_v1' for seed in SEEDS]
                for method in ('Native','Static','Dynamic')}
            last=0.
            while not all(completed(n) for group in names.values() for n in group):
                if time.time()>cutoff-900 or (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
                    write(ROOT/f'state/not_run_conditional_O64_q{qps:g}_matched.json',dict(
                        status='NOT_RUN',reason='Insufficient remaining admission time for three matched starts. Do not infer a matched-policy result.'))
                    return
                if time.time()-last>60:
                    last=time.time();print('WAIT_CONDITIONAL_O64_MATCHING',qps,flush=True)
                time.sleep(15)
            values={m:[metrics(n) for n in group] for m,group in names.items()}
            ni={key:ratio_interval([r[key] for r in values['Native']],
                [r[key] for r in values['Dynamic']]) for key in ('ITL_p99_ms','TPOT_p95_ms')}
            rate=ratio_interval([r['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'] for r in values['Static']],
                [r['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'] for r in values['Dynamic']])
            varying=any(len(r['actual_cut_histogram'])>1 for r in values['Dynamic'])
            migrated=all(r['work_accounting']['completed_migrated_requests']>0 for r in values['Dynamic'])
            passed=varying and migrated and all(r['ratio_CI95'] and r['ratio_CI95'][1]<=1.05 for r in ni.values()) and rate['ratio_CI95'] and rate['ratio_CI95'][0]>1
            write(ROOT/f'analysis/O64_q{qps:g}_conditional_matched_admission.json',dict(
                Native_relative=ni,Dynamic_over_Static_SLO_rate=rate,admit=bool(passed),
                actual_cut_variation=varying,actual_migration_every_start=migrated,
                evidence='TRACE_ANALYSIS',scope='This gate selects only additional diagnostic controls. It does not retune a policy, alter the frozen confirmation, replace a failed seed, or qualify a matched claim without actual matching.'))
            if not passed:
                continue
            entries=[]
            for seed,source in zip(SEEDS,names['Dynamic']):
                name=f'heldout_histogram_geometry_control_N128_O64_q{qps:g}_s{seed}_v1'
                cfg=control_config(name,source,seed)
                if not execute(save(cfg)):
                    continue
                gate=compare_geometry(source,name)
                entries.append(dict(source=source,name=name,gate=gate,metrics=metrics(name)))
                write(ROOT/f'analysis/heldout_O64_q{qps:g}_exact_histogram_geometry_controls.json',entries)
            if entries:
                summarize(entries,ROOT/f'analysis/heldout_O64_q{qps:g}_policy_matched_actual_geometry_statistics.json')
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
