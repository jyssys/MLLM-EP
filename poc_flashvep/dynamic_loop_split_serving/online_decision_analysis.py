"""Separate online proposals, frontier rejection, and completed D work."""
from collections import Counter
import csv
from .common import *
from .work_accounting import measured_partitions


def evaluate(cfg, decisions, partitions, completed, begin):
    events = [row for row in decisions if row.get('online') is True
              and row['time'] >= begin and row['request_id'].startswith('r0_')]
    admissions = [row for row in events if row['event'] == 'ADMISSION_CALIBRATED_PLAN']
    frontiers = [row for row in events if row['event'] == 'FRONTIER_CALIBRATED_RECHECK']
    expected_ids = {f'r0_q{index}' for index in range(cfg['count'])}
    assert len(admissions) == cfg['count']
    assert {row['request_id'] for row in admissions} == expected_ids
    full = cfg['depth'] if cfg['axis'] == 'loop' else cfg['length']
    done = [row for seed, row in partitions.items() if seed in completed]
    assert len(done) == cfg['count']
    partial = [row for row in done if row['partial']]
    accepted = sum(row['reason'] == 'CALIBRATED_FRONTIER_ACCEPT' for row in frontiers)
    assert len(partial) == accepted, 'Online acceptance must equal completed partial work'
    planned = sum(row['selected_key'] != full for row in admissions)
    rejected = sum(row['reason'] == 'SAFE_FULL_PD_FALLBACK' for row in frontiers)
    assert planned == accepted + rejected
    return dict(name=cfg['name'], model=cfg['model'], axis=cfg['axis'], phase=cfg['phase'],
        offered=cfg['count'], completed=len(done), partial_plans=planned,
        frontier_accepts=accepted, frontier_rejects=rejected,
        completed_offloaded_serving_requests=len(partial),
        completed_offload_fraction=len(partial) / cfg['count'],
        completed_Full_PD_requests=len(done) - len(partial),
        assigned_D_recurrent_rows=sum(row['assigned_D_recurrent_token_rows'] for row in partial),
        accepted_partial_keys=dict(Counter(str(row['accepted_cut'] if cfg['axis'] == 'loop'
            else row['accepted_prefix_tokens']) for row in partial)),
        admission_reasons=dict(Counter(row['reason'] for row in admissions)),
        frontier_reasons=dict(Counter(row['reason'] for row in frontiers)),
        scope='Actual unforced online decisions. Forced branch tests and startup requests are excluded. '
              'N32 heavy state probes are correctness evidence, not clean performance. Rows are not FLOPs.')


def main():
    initialize()
    results = []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if not isinstance(cfg, dict) or not cfg.get('dynamic'):
            continue
        if cfg.get('phase') == 'EXACT_DYNAMIC_BRANCH_DIAGNOSTIC':
            continue
        folder = ROOT / 'raw' / cfg['name']
        receipt = folder / 'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        begin = json.loads((folder / 'measurement_begin.json').read_text())['time']
        expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
        parts = measured_partitions([json.loads(line)
            for source in folder.glob('accepted_partitions_*.jsonl')
            for line in source.read_text().splitlines()], expected, begin)
        decisions = [json.loads(line) for line in (folder / 'online_decisions.jsonl').read_text().splitlines()]
        completed = {row['seed'] for line in (folder / 'requests.jsonl').read_text().splitlines()
            if not (row := json.loads(line)).get('warmup')}
        result = evaluate(cfg, decisions, parts, completed, begin)
        write_derived(folder / 'online_decision_summary.json', result)
        results.append(result)
    write_derived(ROOT / 'state/online_decision_summary.json', dict(results=results))
    if results:
        columns = sorted({key for row in results for key in row if not isinstance(row[key], dict)})
        with (ROOT / 'online_decision_summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('actual_unforced_online_decisions'):
        main()
