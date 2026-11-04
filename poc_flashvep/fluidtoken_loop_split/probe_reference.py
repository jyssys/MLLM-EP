"""Compare identical retained probe inputs against a qualified native reference."""
from .common import *


def verify(name):
    folder = ROOT / 'raw' / name
    reference = 'recovery_huginn_original_clean384_exact_probe_reference'
    ref_folder = ROOT / 'raw' / reference
    certified = json.loads((ref_folder / 'recovery_gate.json').read_text())
    assert certified['passed']
    probe_file = folder / 'D_probe_requests.jsonl'
    rows = [json.loads(line) for line in probe_file.read_text().splitlines()] if probe_file.exists() else []
    refs = [json.loads(line) for line in (ref_folder / 'requests.jsonl').read_text().splitlines()]
    refs = {row['seed']: row for row in refs if not row.get('warmup')}
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    requests = [row for row in requests if not row.get('warmup')]
    failures = [row['seed'] for row in rows if row['seed'] not in refs or
                any(row[key] != refs[row['seed']][key] for key in ('input_ids', 'output_ids', 'exit_depths'))]
    before = json.loads((folder / 'before_measured.json').read_text())
    after = json.loads((folder / 'after_measured.json').read_text())
    observed = sum(row['issued_loop_rows'] - before[key]['issued_loop_rows'] for key, row in after.items())
    expected = sum((row['input_tokens'] + row['output_tokens'] - 1) * cfg['depth'] for row in requests)
    expected += sum(len(row['input_ids']) * cfg['depth'] for row in rows)
    result = dict(name=name, reference=reference,
        reference_SHA256=digest(ref_folder / 'requests.jsonl'),
        completed_probes=len(rows), expected_probes=cfg['D_probe_count'], failures=failures,
        exact=not failures and len(rows) == cfg['D_probe_count'] and len({row['seed'] for row in rows}) == len(rows),
        expected_recurrent_rows=expected, observed_recurrent_rows=observed, full_work=observed == expected,
        scope='Exact probe inputs, seeds, outputs and depth against the qualified native reference. Not migrated continuation or original-HF fidelity.')
    target = folder / 'retained_native_probe_correctness.json'
    if target.exists():
        assert json.loads(target.read_text()) == result
    else:
        write(target, result)
    return result


def completion_check(rows, enqueued, receipt, upper_limit):
    """The admission cap is not the number of requests actually offered to D."""
    completed_ids = [row['request_id'] for row in rows]
    enqueued_ids = [row['request_id'] for row in enqueued]
    return (0 < len(enqueued_ids) <= upper_limit
            and len(set(enqueued_ids)) == len(enqueued_ids)
            and len(set(completed_ids)) == len(completed_ids)
            and set(completed_ids) == set(enqueued_ids)
            and receipt.get('enqueued') == len(enqueued_ids)
            and receipt.get('completed') == len(completed_ids)
            and receipt.get('all_completed') is True)


def verify_admitted(name):
    # Preserve the original quota-based result and its verdict.
    old = verify(name)
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    request_file, event_file = folder / 'D_probe_requests.jsonl', folder / 'D_probe_events.jsonl'
    rows = [json.loads(line) for line in request_file.read_text().splitlines()] if request_file.exists() else []
    enqueued = [json.loads(line) for line in event_file.read_text().splitlines()] if event_file.exists() else []
    enqueued = [row for row in enqueued if row['event'] == 'ENQUEUED']
    receipt_file = folder / 'D_probe_completion.json'
    receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
    complete = completion_check(rows, enqueued, receipt, cfg['D_probe_count'])
    admitted = {row['request_id']: row for row in enqueued}
    identity = all(row['request_id'] in admitted and all(row[key] == admitted[row['request_id']][key]
                  for key in ('seed', 'input_ids')) for row in rows)
    result = dict(old, original_quota_verdict=old['exact'], maximum_probe_admissions=cfg['D_probe_count'],
        actual_probe_admissions=len(enqueued), all_admitted_completed=complete,
        exact=complete and identity and not old['failures'] and len({row['seed'] for row in rows}) == len(rows),
        completion_contract='ADMITTED_EXACT_V2',
        scope='Every admitted probe must finish with exact native output and full depth. Unadmitted capacity is not an output mismatch. The quota result remains separate.')
    write(folder / 'admitted_native_probe_correctness_v2.json', result)
    return result
