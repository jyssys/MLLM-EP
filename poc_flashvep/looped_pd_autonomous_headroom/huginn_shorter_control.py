"""Test lower KV demand with a new, independently calibrated input length."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .campaign import gate
from .analysis_lane import quiet_analysis
from .resume_joint_review import analyze_capture
from .verify_long_probes import verify
from .independent_joint_repeats import captures


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    from ..looped_pd_headroom_recovery import primary_campaign as calibration
    calibration.ROOT = ROOT
    base = json.loads((ROOT/'configs/window_huginn_s1_baseline384.json').read_text())
    base.update(length=2048, state_capacity_tokens=2048, chunk=2048, token_budget=4096,
        P_prefill_graph_rows=4096, cap=16, workload='L2048_O16', timeout=1200,
        data_scope=dict(scope='Pinned public tokenizer prefixes. Explicit controlled generation.'))
    ledger = []
    def run(cfg, count, rate=None, split='test'):
        name = put(cfg, count, rate, split)
        write(ROOT/'state/huginn_shorter.json', dict(stage='RUNNING', name=name, time=time.time()))
        return name, execute(name)
    # Differential native-state gate before the new long geometry enters serving.
    refs = []
    for label, graph in (('eager', 0), ('graph', 4096)):
        cfg = copy.deepcopy(base)
        cfg.update(name=f'short_huginn_{label}_gate8', P_prefill_graph_rows=graph,
            phase='correctness', verify_transfer=True, state_probe=True)
        name, okay = run(cfg, 8, 1.)
        assert okay and gate(name), f'State/work gate failed: {name}'
        refs.append(name)
    assert evidence.compare(*refs)['passed'], 'Eager/graph output mismatch'
    refs32 = []
    for label, graph in (('eager', 0), ('graph', 4096)):
        cfg = copy.deepcopy(base)
        cfg.update(name=f'short_huginn_{label}_independent32', P_prefill_graph_rows=graph,
            phase='correctness', arrival_seed=2026111401)
        name, okay = run(cfg, 32, 1., 'confirmation')
        assert okay and evidence.full_work(name)['passed']
        refs32.append(name)
    assert evidence.compare(*refs32)['passed'], 'Independent output mismatch'
    record = dict(model='huginn', length=2048, calibration=[])
    for chunk in (1024, 2048):
        cfg = copy.deepcopy(base)
        cfg.update(name=f'short_huginn_C{chunk}_serial16', chunk=chunk,
            phase='calibration', calibration_serial=True)
        name, okay = run(cfg, 16, None, 'calibration')
        assert okay
        record['calibration'].append(name)
    assert calibration.freeze_slo(record), 'New-length calibration failed'
    write(ROOT/'state/huginn_L2048_calibration_clarification.json', dict(
        calibration_runs=record['calibration'], actual_chunks=[1024, 2048],
        note='The inherited rule label says Atomic/C2048. The recorded runs use C1024 and full C2048. Numeric frozen thresholds are unchanged.'))
    write(ROOT/'state/capacity_short_huginn_L2048_invalid_dispatch.json', dict(
        status='INVALID_CAPACITY_DISPATCH',
        reason='The original configuration selected the serving worker. Its two-request output is not an independent capacity measurement.',
        replacement='capacity_short_huginn_L2048_v2'))
    cfg = copy.deepcopy(base)
    cfg.update(name='capacity_short_huginn_L2048_v2', worker='capacity_worker', capacity_mode='P_ONLY',
        P_graph_local_capacity=True, jobs=[dict(batch=b, cap=1, repeats=3) for b in (1, 2)])
    name, okay = run(cfg, 2, None, 'calibration')
    assert okay
    cohorts = [json.loads(s) for s in (ROOT/'raw'/name/'cohorts.jsonl').read_text().splitlines()]
    reference = max(r['job']['batch']/r['prefill_host_service_s'] for r in cohorts if r.get('status') == 'COMPLETE')
    candidates = []
    for factor in (1.1, 1.3):
        cfg = copy.deepcopy(base)
        rate = round(reference*factor, 3)
        cfg.update(name=f'short_huginn_factor{int(100*factor)}_pilot24', phase='qps_pilot')
        name, okay = run(cfg, 24, rate)
        assert okay and evidence.full_work(name)['passed']
        with quiet_analysis(name+'_metrics'):
            metrics, _, _ = serving.serving_one(cfg)
        ledger.append(dict(name=name, pilot=metrics, reference=reference))
        if metrics['request_TPOT_ms_p95'] <= .9*metrics['TPOT_threshold_ms'] and metrics['token_ITL_ms_p99'] <= metrics['ITL_threshold_ms']:
            candidates.append((rate, cfg))
    write(ROOT/'state/huginn_shorter_results.json', ledger)
    if not candidates:
        write(ROOT/'state/huginn_shorter.json', dict(stage='NO_D_SLO_CANDIDATE', time=time.time()))
        return
    rate, selected = max(candidates, key=lambda value: value[0])
    original = json.loads((ROOT/'data/confirmation_huginn_L2048.json').read_text())
    write(ROOT/'data/short_confirmation_huginn_L2048.json', [copy.deepcopy(original[i % len(original)]) for i in range(384)])
    for start, pair in enumerate(((4, 5), (6, 7), (4, 5)), 1):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(base)
        cfg.update(name=f'short_huginn_s{start}_baseline384', phase='final_repeat',
            arrival_seed=2026111500+start, independent_start=start, gpus=list(pair),
            data_scope=dict(original_pool_records=128, offered_count=384, explicit_cyclic_reuse=True))
        baseline, okay = run(cfg, 384, rate, 'short_confirmation')
        assert okay
        probe = copy.deepcopy(cfg)
        probe.update(name=f'short_huginn_s{start}_probe384', D_probe_tokens=16,
            D_probe_count=12, D_probe_interval_s=4.)
        clean, okay = run(probe, 384, rate, 'short_confirmation')
        assert okay and evidence.compare(baseline, clean)['passed']
        correctness = verify(clean)
        assert correctness.get('exact') and correctness.get('full_work')
        prof = copy.deepcopy(probe)
        prof.update(name=f'short_huginn_s{start}_nsys384', matched_clean=clean,
            P_only_profiler=True, nsys=True, cuda_only_profile=False, capture_role='prefill',
            capture_seconds=0., autonomous_capture_seconds=8., capture_arrival_fraction=.35,
            capture_prewarmed=False, capture_until_close=False, profile_stop_at_exit=False,
            nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
        name, okay = run(prof, 384, rate, 'short_confirmation')
        row = dict(model='huginn', start=start, baseline=baseline, status='PROFILE_FAILED')
        if okay:
            row.update(analyze_capture(prof, clean))
            row['measured_captures'] = {n: captures(n) for n in (baseline, clean, name)}
            row['accepted_joint_screen'] = row['joint_screen_pass'] and not any(row['measured_captures'].values())
        ledger.append(row)
        write(ROOT/'state/huginn_shorter_results.json', ledger)
        if not row.get('accepted_joint_screen'):
            break
    write(ROOT/'state/huginn_shorter.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/huginn_shorter.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
