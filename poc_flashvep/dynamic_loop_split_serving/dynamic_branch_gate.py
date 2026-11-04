"""Require the intended diagnostic branch, not only identical final output."""
from .common import *


def branch_reached(axis, case, key, count, work, rejection_events):
    requests = work['requests']
    if len(requests) != count or work['completed'] != count:
        return False
    if case == 'accept':
        field = 'cut' if axis == 'loop' else 'prefix'
        return (work['completed_offloaded_serving_requests'] == count
                and work['measured_D_prompt_rows_from_active_counter'] > 0
                and all(row['partial'] and row[field] == key for row in requests))
    if case != 'reject':
        raise ValueError('Unknown exact-branch diagnostic')
    return (work['completed_offloaded_serving_requests'] == 0
            and work['measured_D_prompt_rows_from_active_counter'] == 0
            and not any(row['partial'] for row in requests)
            and rejection_events == count)


def check(name, case, key):
    from .work_accounting import summarize
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    folder = ROOT / 'raw' / name
    start = json.loads((folder / 'measurement_begin.json').read_text())['time']
    events = [json.loads(line) for line in (folder / 'online_decisions.jsonl').read_text().splitlines()]
    rejections = sum(row['event'] == 'FORCED_EXACT_FRONTIER_REJECTION' and row['time'] >= start
                     for row in events)
    work = summarize(name)
    result = dict(name=name, case=case, requested_key=key, offered=cfg['count'],
        calibration_serial=cfg.get('calibration_serial', False),
        completed_offloaded_serving_requests=work['completed_offloaded_serving_requests'],
        actual_D_prompt_rows=work['measured_D_prompt_rows_from_active_counter'],
        measured_forced_frontier_rejections=rejections,
        passed=branch_reached(cfg['axis'], case, key, cfg['count'], work, rejections),
        scope='Forced-branch correctness diagnostic. The configuration records serial or concurrent execution. Identical final output alone does not prove that the requested branch executed. This is not online performance.')
    write_derived(folder / 'dynamic_branch_gate.json', result)
    return result
