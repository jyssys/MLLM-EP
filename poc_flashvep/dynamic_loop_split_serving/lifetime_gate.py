"""Check recorded ACK ownership and final allocation retirement."""
from .common import *


def evaluate(closed, events, owners, contracts, count):
    measured = [row for row in closed['metrics'] if row['request_id'].startswith('r0_')]
    failures = []
    if len(measured) != count or any(row.get('cancelled') for row in measured):
        failures.append('Incomplete or cancelled measured population')
    for row in measured:
        tid = row['transfer_id']
        records = [record for record in events if record.get('tid') == tid]
        released = {record['role'] for record in records if record['event'] == 'COORDINATOR_released'}
        ack = [record for record in owners if record.get('tid') == tid and record['role'] == 'prefill'
               and record['event'] == 'NATIVE_ACK_RECEIVED' and record.get('acknowledged') is True]
        forgotten = [record for record in owners if record.get('tid') == tid and record['role'] == 'prefill'
                     and record['event'] == 'NATIVE_CONNECTOR_FORGET']
        if released != {'prefill', 'decode'} or not ack or not forgotten:
            failures.append(dict(request_id=row['request_id'], reason='Missing release or source ACK evidence'))
            continue
        if min(record['time'] for record in forgotten) < min(record['time'] for record in ack):
            failures.append(dict(request_id=row['request_id'], reason='Source retirement precedes recorded ACK'))
        if any(record.get('allocation_after') for record in forgotten):
            failures.append(dict(request_id=row['request_id'], reason='Allocation remains after source retirement'))
    if {row['role'] for row in contracts} != {'prefill', 'decode'}:
        failures.append('Missing final native contract')
    elif any(row['used_KV_blocks'] != 0 or row['prefix_caching_enabled'] or row['reused_prefix_tokens'] != 0
             for row in contracts):
        failures.append('Final allocation or prefix-reuse contract failed')
    return dict(passed=not failures, compared=len(measured), failures=failures,
        scope='Recorded native ACK precedes source retirement. Both coordinator credits retire and final native KV allocations are empty. This is not GPU DMA timing or HF fidelity.')


def check(name):
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    events = [json.loads(line) for line in (folder / 'lifetime_events.jsonl').read_text().splitlines()]
    owners = [json.loads(line) for path in folder.glob('lifetime_owner_*.jsonl') for line in path.read_text().splitlines()]
    contracts = [json.loads(path.read_text()) for path in folder.glob('native_contract_final_*.json')]
    result = evaluate(json.loads((folder / 'closed_stats.json').read_text()), events, owners, contracts, cfg['count'])
    write_derived(folder / 'native_lifetime_gate.json', result)
    return result


if __name__ == '__main__':
    import sys
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('native_ACK_and_allocation_audit'):
        print(json.dumps({name: check(name) for name in sys.argv[1:]}, indent=2))
