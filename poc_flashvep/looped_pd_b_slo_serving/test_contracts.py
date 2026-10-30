"""CPU-only checks for arrival and SLO semantics."""
import json
from pathlib import Path
import pytest

from .worker import arrival_trace


def test_frozen_arrival_trace_is_identical_across_arms():
    first, end = arrival_trace(200, 2., 2026103001)
    other, other_end = arrival_trace(200, 2., 2026103001)
    assert first == other and end == other_end
    assert len(first) == 200 and first[0] > 0 and end > first[-1]
    assert all(x < y for x, y in zip(first, first[1:]))


def test_independent_starts_use_distinct_arrival_traces():
    assert arrival_trace(20, 1., 1) != arrival_trace(20, 1., 2)


def test_repair_feedback_rejects_partial_or_inexact_cohorts(monkeypatch):
    from . import repair_feedback
    monkeypatch.setattr(repair_feedback, 'parity', lambda a, b: dict(N=7, exact=7, complete_ids=True))
    assert repair_feedback.check('left', 'right', 7)['passed']
    assert not repair_feedback.check('left', 'right', 8)['passed']
    monkeypatch.setattr(repair_feedback, 'parity', lambda a, b: dict(N=7, exact=6, complete_ids=True))
    assert not repair_feedback.check('left', 'right', 7)['passed']
    monkeypatch.setattr(repair_feedback, 'parity', lambda a, b: dict(N=7, exact=7, complete_ids=False))
    assert not repair_feedback.check('left', 'right', 7)['passed']


def test_repair_epoch_does_not_pool_into_original_capacity(monkeypatch, tmp_path):
    import csv
    from . import analyze
    monkeypatch.setattr(analyze, 'ROOT', tmp_path)
    (tmp_path / 'configs/jobs').mkdir(parents=True)
    for epoch in ('original', 'canonical_core_linear_bf16_v1'):
        job = dict(name=epoch, model='ouro', arm='PD_G1', workload='W1', dataset='W1',
                   count=1, rate=1., seed=1, gpus=[0, 1], execution_epoch=epoch)
        (tmp_path / 'configs/jobs' / (epoch + '.json')).write_text(json.dumps(job))
        folder = tmp_path / 'raw' / epoch
        folder.mkdir(parents=True)
        (folder / 'status.json').write_text('{"status":"COMPLETE"}')
    def summarize(job):
        row = dict(run=job['name'], model='ouro', arm='PD_G1', GPU_count=2,
                   physical_GPUs=[0, 1], workload='W1', status='COMPLETE',
                   diagnostic=False, execution_epoch=job['execution_epoch'])
        return dict(row), dict(row), dict(row)
    monkeypatch.setattr(analyze, 'summarize', summarize)
    analyze.main()
    original = list(csv.DictReader((tmp_path / '03_primary_w1_capacity.csv').open()))
    repaired = list(csv.DictReader((tmp_path / 'repair_epoch_w1_capacity.csv').open()))
    assert len(original) == len(repaired) == 1
    assert original[0]['execution_epoch'] == 'original'
    assert repaired[0]['execution_epoch'] == 'canonical_core_linear_bf16_v1'


def test_burst_arrivals_are_a_distinct_preregistered_trace():
    a, end_a = arrival_trace(200, 1., 9)
    b, end_b = arrival_trace(200, 1., 9, burst=True)
    assert a != b and end_b < end_a
    assert all(x < y for x, y in zip(b, b[1:]))


def test_dataset_calibration_and_heldout_are_disjoint():
    from .common import ROOT
    groups = [json.loads((ROOT / 'data' / (name + '.json')).read_text()) for name in ('calibration', 'correctness', 'W1')]
    hashes = [{row['prompt_sha256'] for row in group} for group in groups]
    assert all(not hashes[i] & hashes[j] for i in range(3) for j in range(i))
    assert len(hashes[2]) == 200
    assert all(max(row['input_lengths'].values()) + 256 <= 4096 for group in groups for row in group)


def test_one_token_TPOT_convention_and_arrival_denominator(monkeypatch, tmp_path):
    from . import analyze, jobs
    monkeypatch.setattr(analyze, 'ROOT', tmp_path)
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    folder = tmp_path / 'raw/example'
    folder.mkdir(parents=True)
    (tmp_path / 'configs').mkdir()
    (folder / 'status.json').write_text(json.dumps(dict(status='COMPLETE')))
    records = [dict(sample_id=i, output_tokens=1, input_tokens=100, ttft_ms=10, tpot_ms=0,
                    e2e_ms=10, finish_reason=('eos', 'stop', 'length')[i], finish=.1, token_events=[], warmup=False) for i in range(3)]
    (folder / 'requests.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (folder / 'window.json').write_text(json.dumps(dict(start=0, window_seconds=10, evaluation_end=10,
                                                      actual_span_seconds=15, drain_seconds=5, actual_trace_rate=.4)))
    (tmp_path / 'configs/slo_lock.json').write_text(json.dumps(dict(ouro=dict(TTFT_ms=100, TPOT_ms=1, secondary_ITL_ms=5))))
    job = dict(name='example', model='ouro', arm='B', workload='W1', gpus=[0,1], seed=1, rate=.4, count=4, cap=128)
    row, _, _ = analyze.summarize(job)
    assert row['qualified_requests'] == 3 and row['joint_attainment_percent'] == 75
    assert row['qualified_requests_s_window'] == pytest.approx(.3)
    assert row['qualified_requests_s_with_drain'] == pytest.approx(.2)
    assert row['planned_but_incomplete_fraction'] == .25
    assert row['EOS_stops'] == 2 and row['cap_stops'] == 1


def test_output_failure_cannot_become_accepted_capacity(monkeypatch, tmp_path):
    from . import finish, analyze
    monkeypatch.setattr(finish, 'ROOT', tmp_path)
    monkeypatch.setattr(analyze, 'ROOT', tmp_path)
    (tmp_path / 'state').mkdir()
    (tmp_path / 'state/ouro_B_serving_parity_failure.json').write_text('{}')
    rows = [dict(model='ouro', GPU_count='2', arm='B', offered_rate='1', seed='1',
                 status='COMPLETE', joint_attainment_percent='95', weak_queue_check='True',
                 completed_throughput_with_drain='1', actual_trace_rate='1')]
    result = next(row for row in finish.capacity_bounds(rows)
                  if row['model'] == 'ouro' and row['GPU_count'] == 2 and row['arm'] == 'B')
    assert result['lower_tested_rate'] == 1
    assert result['accepted_lossless_capacity'] == 'REJECTED_CORRECTNESS_GATE'


def test_profile_launcher_import_does_not_start_workers():
    import importlib
    from ..looped_pd_abc import worker as reused
    before = reused.main
    launcher = importlib.import_module('poc_flashvep.looped_pd_b_slo_serving.profile_python')
    assert callable(launcher.main)
    assert reused.main is before


def test_failed_trace_keeps_all_planned_arrivals_and_unknown_drain(monkeypatch, tmp_path):
    from . import analyze, jobs
    monkeypatch.setattr(analyze, 'ROOT', tmp_path)
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    folder = tmp_path / 'raw/failed'
    folder.mkdir(parents=True)
    (tmp_path / 'configs').mkdir()
    (folder / 'status.json').write_text(json.dumps(dict(status='FAILED', error='Unit-test timeout')))
    row = dict(sample_id=0, output_tokens=1, input_tokens=10, ttft_ms=10, tpot_ms=0,
               e2e_ms=100, finish_reason='stop', arrival=10, finish=10.1, token_events=[], warmup=False)
    (folder / 'requests.jsonl').write_text(json.dumps(row))
    (folder / 'arrival_trace.json').write_text(json.dumps(dict(offsets=[1,2,3,4], window_seconds=5)))
    (tmp_path / 'configs/slo_lock.json').write_text(json.dumps(dict(ouro=dict(TTFT_ms=100, TPOT_ms=1, secondary_ITL_ms=5))))
    job = dict(name='failed', model='ouro', arm='PD_G1', workload='W1', gpus=[0,1], seed=1, rate=1, count=4, cap=128)
    value, percentiles, _ = analyze.summarize(job)
    assert value['qualified_requests'] == 1 and value['joint_attainment_percent'] == 25
    assert value['planned_but_incomplete_fraction'] == .75
    assert value['drain_seconds'] == value['qualified_requests_s_with_drain'] == 'NOT_MEASURED'
    assert percentiles['p99_scope'] == 'FAILED_OR_INCOMPLETE_RUN_SURVIVORS_ONLY'
