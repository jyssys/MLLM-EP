"""Report native host-message intervals without calling them GPU DMA time."""
from .common import json
from .simulator import percentile

INTERVALS = {
    'P_complete_receipt_to_commit_receipt_ms': ('complete', 'commit'),
    'commit_receipt_to_received_receipt_ms': ('commit', 'received'),
    'received_receipt_to_P_ACK_ms': ('received', 'ACK'),
    'P_ACK_to_P_forget_receipt_ms': ('ACK', 'forget'),
    'P_complete_receipt_to_P_credit_release_ms': ('complete', 'released'),
}
CONTINUATION_GAPS = ('received_receipt_to_D_prompt_entry_ms',
    'P_model_stream_end_to_D_prompt_entry_ms', 'D_prompt_end_to_first_token_ms')


def continuation_gaps(record, received):
    missing = dict.fromkeys(CONTINUATION_GAPS)
    if record is None or record.get('D_continuation_to_first_token_ms') is None:
        return missing
    duration = record['D_continuation_to_first_token_ms']
    d_start = record['first_token'] - duration / 1000
    return dict(received_receipt_to_D_prompt_entry_ms=1000 * (d_start - received)
                    if received is not None else None,
        P_model_stream_end_to_D_prompt_entry_ms=1000 * (d_start - record['actual_P_model_stream_entry']['time'])
                    - record['P_prompt_device_enclosure_ms'],
        D_prompt_end_to_first_token_ms=duration - record['D_prompt_device_enclosure_ms'])


def intervals(events, owners, since, count):
    measured = {row['tid']: row['request_id'] for row in events
                if row['event'] == 'COORDINATOR_prefill_started' and row['time'] >= since}
    assert len(measured) == count, 'Measured native lifecycle coverage is incomplete'
    stamps = {tid: {} for tid in measured}
    names = {'COORDINATOR_prefill_complete': 'complete', 'COORDINATOR_commit': 'commit',
             'COORDINATOR_received': 'received'}
    for row in events:
        if row['tid'] not in stamps:
            continue
        key = names.get(row['event'])
        if row['event'] == 'COORDINATOR_released' and row['role'] == 'prefill':
            key = 'released'
        if key is not None:
            stamps[row['tid']][key] = row['time']
    for row in owners:
        if row['tid'] not in stamps or row['role'] != 'prefill':
            continue
        key = {'NATIVE_ACK_RECEIVED': 'ACK', 'NATIVE_CONNECTOR_FORGET': 'forget'}.get(row['event'])
        if key is not None:
            stamps[row['tid']][key] = row['time']
    rows = []
    for tid, times in stamps.items():
        values = {name: 1000 * (times[right] - times[left])
                  if left in times and right in times else None
                  for name, (left, right) in INTERVALS.items()}
        rows.append(dict(tid=tid, request_id=measured[tid], received_receipt_time=times.get('received'), **values))
    return rows


def analyze(folder, count):
    events = [json.loads(line) for line in (folder / 'lifetime_events.jsonl').read_text().splitlines()]
    owners = [json.loads(line) for path in folder.glob('lifetime_owner_*.jsonl')
              for line in path.read_text().splitlines()]
    since = json.loads((folder / 'measurement_begin.json').read_text())['time']
    rows = intervals(events, owners, since, count)
    source = folder / 'service_latency_requests.json'
    service = {row['request_id']: row for row in json.loads(source.read_text())} if source.exists() else {}
    for row in rows:
        row.update(continuation_gaps(service.get(row['request_id']), row['received_receipt_time']))
    summary = dict(requests=count)
    for field in tuple(INTERVALS) + CONTINUATION_GAPS:
        values = [row[field] for row in rows if row[field] is not None]
        summary[field + '_coverage'] = len(values)
        summary[field + '_negative_intervals'] = sum(value < 0 for value in values)
        for q in (.5, .95, .99):
            summary[field + f'_p{int(q * 100)}'] = percentile(values, q) if values else None
    return rows, summary
