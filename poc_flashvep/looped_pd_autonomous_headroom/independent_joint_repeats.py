"""Repeat the Ouro joint screen with independent starts and complete warm-up."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis
from .verify_long_probes import verify
from .resume_joint_review import analyze_capture


def captures(name):
    return [dict(row, log=p.name) for p in (ROOT/'raw'/name).glob('graph_captures_*.jsonl')
            for row in map(json.loads, p.read_text().splitlines()) if row.get('measured_phase')]


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    template = json.loads((ROOT/'configs/joint_confirm_ouro_s1_n128_probe.json').read_text())
    ledger = []
    for start, pair in enumerate(((0, 1), (2, 3), (0, 1)), 1):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(template)
        cfg.update(name=f'qualified_ouro_s{start}_baseline128', warmup_count=3,
            arrival_seed=2026110900+start, independent_start=start, gpus=list(pair),
            GPU_pair_crossover=True, phase='final_repeat')
        cfg['native_transport_limits'] = dict(cfg.get('native_transport_limits', {}), shutdown_timeout=300)
        for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
            cfg.pop(key, None)
        baseline = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
        row = dict(model='ouro', start=start, baseline=baseline)
        write(ROOT/'state/independent_joint_repeats.json', dict(stage='BASELINE', name=baseline, time=time.time()))
        if not execute(baseline) or not evidence.full_work(baseline)['passed']:
            row['status'] = 'BLOCKED_BASELINE'
            ledger.append(row)
            break
        row['baseline_measured_captures'] = captures(baseline)
        with quiet_analysis(baseline+'_metrics'):
            metrics, _, _ = serving.serving_one(cfg)
        row['baseline_metrics'] = metrics
        if metrics['request_TPOT_ms_p95'] > .9*metrics['TPOT_threshold_ms'] or metrics['token_ITL_ms_p99'] > metrics['ITL_threshold_ms']:
            row['status'] = 'BASELINE_D_SLO_FAILED'
            ledger.append(row)
            break
        probe = copy.deepcopy(cfg)
        probe.update(name=f'qualified_ouro_s{start}_probe128', D_probe_tokens=16,
            D_probe_count=8, D_probe_interval_s=2.)
        clean = put(probe, 128, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/independent_joint_repeats.json', dict(stage='PROBE', name=clean, time=time.time()))
        if not execute(clean) or not evidence.compare(baseline, clean)['passed']:
            row['status'] = 'BLOCKED_PROBE_OR_PARITY'
            ledger.append(row)
            break
        row['probe_correctness'] = verify(clean)
        row['clean_measured_captures'] = captures(clean)
        if not all(row['probe_correctness'].get(k) for k in ('exact', 'full_work')):
            row['status'] = 'BLOCKED_PROBE_CORRECTNESS'
            ledger.append(row)
            break
        prof = copy.deepcopy(probe)
        prof.update(name=f'qualified_ouro_s{start}_nsys128', matched_clean=clean,
            P_only_profiler=True, nsys=True, cuda_only_profile=False,
            bounded_native_capture_seconds=0., capture_role='prefill', capture_seconds=3.,
            capture_prewarmed=True, capture_until_close=True, profile_stop_at_exit=True,
            nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
        name = put(prof, 128, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/independent_joint_repeats.json', dict(stage='PROFILE', name=name, time=time.time()))
        if execute(name):
            row.update(analyze_capture(prof, clean))
            row['profile_measured_captures'] = captures(name)
            row['fully_warmed'] = not any(row[k] for k in (
                'baseline_measured_captures', 'clean_measured_captures', 'profile_measured_captures'))
            row['accepted_joint_screen'] = row['joint_screen_pass'] and row['fully_warmed']
        else:
            row['status'] = 'PROFILE_FAILED'
        ledger.append(row)
        write(ROOT/'state/independent_joint_repeats_results.json', ledger)
        # Do not discard a failed independent seed or replace it with a favorable seed.
        # Complete the predeclared three starts unless correctness or budget blocks them.
    write(ROOT/'state/independent_joint_repeats_results.json', ledger)
    write(ROOT/'state/independent_joint_repeats.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/independent_joint_repeats.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
