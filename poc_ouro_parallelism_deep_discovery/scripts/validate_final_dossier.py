"""Check publication metadata. These checks do not certify scientific conclusions."""
import argparse
import collections
import datetime as dt
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recorded_duration(rows, cutoff):
    """Sum nonoverlapping coordinator segments. Do not add concurrent workers."""
    require(cutoff.tzinfo is not None, 'Require a timezone-aware cutoff')
    previous = None
    seconds = 0.0
    reported_seconds = 0.0
    segments = []
    for row in rows:
        if row['actor'] != 'coordinator':
            continue
        start = dt.datetime.fromisoformat(row['start_utc'])
        end = dt.datetime.fromisoformat(row['end_utc'])
        require(start.tzinfo is not None and end.tzinfo is not None, 'Require timezone-aware segments')
        require(start <= end <= cutoff, 'Reject reversed or future segments')
        require(previous is None or start >= previous, 'Reject duplicate or overlapping segments')
        duration = (end-start).total_seconds()
        reported = row['elapsed_seconds']
        require(type(reported) in (int,float) and math.isfinite(reported) and reported >= 0,
                'Require finite nonnegative reported seconds')
        places = max(0, -Decimal(str(reported)).as_tuple().exponent)
        require(math.isclose(round(duration, places), reported, rel_tol=0, abs_tol=0.000000001),
                'Recorded segment duration disagrees with its endpoints')
        require(bool(row['evidence']), 'Require evidence for a recorded segment')
        seconds += duration
        reported_seconds += reported
        previous = end
        segments.append(dict(start_utc=start.isoformat(), end_utc=end.isoformat(), seconds=duration,
                             reported_seconds=reported, reported_decimal_places=places,
                             reported_minus_endpoint_seconds=reported-duration))
    return dict(seconds=seconds, hours=seconds/3600, reported_seconds=reported_seconds, segments=segments,
                precision_rule='Check reported seconds at their stored decimal precision. Sum endpoint-derived durations for the completion gate. Preserve the original ledger.',
                limitation='Checks recorded intervals, not whether every interval was substantive. The activity ledger supplies that attestation.')


def local_link_targets(path):
    text = path.read_text()
    targets = []
    for match in re.finditer(r'\[[^\]\n]*\]\(([^)\n]+)\)', text):
        value = match[1].strip()
        if value.startswith('<'):
            require(value.endswith('>'), 'Malformed bracketed Markdown target')
            value = value[1:-1]
        if not value or value.startswith('#') or urlsplit(value).scheme:
            continue
        value = unquote(value.split('#', 1)[0])
        value = re.sub(r':\d+$', '', value)
        target = Path(value)
        if not target.is_absolute():
            target = path.parent/target
        targets.append(target.resolve())
    return targets


def candidate_scope(rows):
    ids = [row['id'] for row in rows]
    require(len(ids) == 19 and set(ids) == {f'C{i:02}' for i in range(1,20)}, 'Require all19 unique candidate IDs')
    for row in rows:
        require(row['candidate_live_E2E'] == 'NOT_RUN', 'Candidate E2E scope changed')
        require(row['candidate_free_running_quality'] == 'NOT_RUN', 'Candidate quality scope changed')
    return dict(count=len(ids), frozen_dispositions=dict(collections.Counter(row['status'] for row in rows)),
                candidate_live_E2E='NOT_RUN', candidate_free_running_quality='NOT_RUN',
                interpretation='Frozen labels are not measured architecture-failure counts.')


def verify_inputs(path):
    value = read(path)
    for relative, digest in value['input_sha256'].items():
        require(sha(ROOT/relative) == digest, 'Changed admitted input: '+relative)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-integrity', type=Path, required=True)
    parser.add_argument('--native-quality', type=Path, required=True)
    parser.add_argument('--cpu-regression', type=Path, required=True)
    parser.add_argument('--staged-regrade', type=Path, required=True)
    parser.add_argument('--allow-incomplete-duration', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a fresh audit destination')
    now = dt.datetime.now(dt.timezone.utc)
    inputs = {}

    def pin(path):
        inputs[str(path.resolve().relative_to(ROOT))] = sha(path)
        return path

    time_path = pin(ROOT/'artifacts/time_ledger.jsonl')
    duration = recorded_duration([json.loads(line) for line in time_path.read_text().splitlines()], now)
    duration_pass = duration['seconds'] >= 36000
    require(duration_pass or args.allow_incomplete_duration, 'Ten substantive coordinator hours are not yet recorded')
    statuses = pin(ROOT/'artifacts/candidate_status.jsonl')
    candidates = candidate_scope([json.loads(line) for line in statuses.read_text().splitlines()])
    hardware = verify_inputs(pin(ROOT/'analysis/native_hybrid_cohorts_complete_v3/summary.json'))
    require(len(hardware['trials']) == 400 and len(hardware['summaries']) == 80,
            'Controlled hardware denominators changed')
    context = verify_inputs(pin(ROOT/'analysis/long_context_complete_v2/summary.json'))
    require(sum(row['clean_trials'] for row in context['runs']) == 144 and len(context['cells']) == 24,
            'Context-control denominators changed')
    staged = verify_inputs(pin(ROOT/'analysis/baseline_extension_complete_v1/summary.json'))
    require(len(staged['records']) == 256, 'Require all four staged64-request arms')
    regrade = verify_inputs(pin(args.staged_regrade))
    unchanged = lambda value: {key: item for key,item in value.items() if key != 'created_utc'}
    require(unchanged(staged) == unchanged(regrade), 'Frozen staged regrade differs beyond its timestamp')
    original = read(pin(ROOT/'artifacts/probe_protocol.json'))
    held_out = {row['id'] for row in original['rows'] if row['campaign_partition'] == 'held_out'}
    require(len(held_out) == 64, 'Require the frozen64-ID original held-out metadata')
    extension = read(pin(ROOT/'artifacts/baseline_extension_protocol_v1.json'))
    extension_ids = {row['id'] for row in extension['rows']}
    require(len(extension_ids) == 64 and not (extension_ids & held_out), 'Extension overlaps original held-out metadata')
    require({row['id'] for row in staged['records']} == extension_ids, 'Staged quality IDs changed')
    integrity = verify_inputs(pin(args.native_integrity))
    require(len(integrity['jobs']) == 2 and all(row['status'] == 'PASS' for row in integrity['jobs']),
            'Both native arms must complete admission')
    native = verify_inputs(pin(args.native_quality))
    require(len(native['records']) == 128, 'Require both native64-request arms')
    require({row['id'] for row in native['records']} == extension_ids, 'Native quality IDs changed')
    require(native['candidate_E2E'] == native['candidate_quality'] == 'NOT_RUN', 'Native baseline is not a candidate result')
    cpu = read(pin(args.cpu_regression))
    require(cpu['status'] == 'PASS' and cpu['failed_modules'] == 0, 'CPU regression did not pass')
    require(cpu['modules'] == len(list((ROOT/'tests').glob('test_*.py'))), 'CPU suite omits a current module')
    require(cpu.get('source_sha256') and cpu.get('source_drift') == [], 'Require a stable CPU package-source receipt')
    for relative,digest in cpu['source_sha256'].items():
        require(sha(ROOT/relative) == digest, 'CPU source changed after regression: '+relative)
    report_paths = [ROOT/'README.md', ROOT/'reports/final_report.md',
        ROOT/'reports/reproduce_final_v1.md', ROOT/'reports/baseline_depth_profile_v1.md',
        ROOT/'reports/native_quality_extension_complete_v1.md',
        ROOT/'reports/recurrent_sampler_camera_ready_supplement_v1.md']
    checked_links = []
    for path in report_paths:
        pin(path)
        for target in local_link_targets(path):
            require(target.exists(), 'Missing report link: '+str(target))
            checked_links.append(dict(report=str(path.relative_to(ROOT)), target=str(target)))
    result = dict(created_utc=now.isoformat(), status='MECHANICAL_CHECKS_PASS',
        duration_pass=duration_pass, recorded_duration=duration, candidates=candidates,
        controlled_trials=400, controlled_cells=80, context_trials=144, context_cells=24,
        staged_quality_records=256, native_quality_records=128, unique_extension_problem_IDs=64,
        staged_regrade='IDENTICAL_EXCEPT_CREATED_UTC',
        original_held_out_metadata_overlap=0, original_held_out_outcomes='NOT_EVALUATED',
        CPU=dict(modules=cpu['modules'], tests=cpu['tests'], status=cpu['status']),
        checked_links=checked_links, input_sha256=inputs,
        limitation='Metadata, hashes and denominators only. No novelty, numerical parity, quality noninferiority, measured candidate speedup, GPU release, or Git transport certification.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status','duration_pass','controlled_trials','context_trials','staged_quality_records','native_quality_records')}))


if __name__ == '__main__':
    main()
