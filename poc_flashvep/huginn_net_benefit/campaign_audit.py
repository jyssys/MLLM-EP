"""Preserve request gates and completion observations for every new serving cell."""
from collections import defaultdict
from .common import *


def main():
    initialize()
    from ..completion_aware_loop_pd import serving_audit as inherited
    inherited.ROOT = ROOT
    inherited.configure = configure
    analysis()
    items, requests, completions = [], [], []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if cfg.get('model') != 'huginn' or cfg.get('count', 0) < 32 or cfg.get('state_probe'):
            continue
        if any(cfg.get(key) for key in ('sparse_device_trace', 'host_cost_trace', 'local_nsys',
                'full_kernel_trace', 'observe_device_execution', 'profile_python')):
            continue
        name = cfg['name']
        receipt = ROOT / 'raw' / name / 'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        item, rows, finished = inherited.inspect(name)
        selected = defaultdict(list)
        begin = json.loads((ROOT/'raw'/name/'measurement_begin.json').read_text())['time']
        for file in (ROOT / 'raw' / name).glob('net_benefit_decisions_*.json'):
            for record in json.loads(file.read_text()).get('records', []):
                if (record['event'] == 'SELECTIVE_CHECKPOINT' and record['time'] >= begin
                        and 0 <= record['seed']-cfg['seed'] < cfg['count']):
                    selected[record['seed']].append(record)
        for row in rows:
            choices = selected[row['sampling_seed']]
            if choices:
                row.update(selective_evaluations=len(choices),
                    selective_exposed=any(choice['target'] < cfg['depth'] for choice in choices),
                    selective_reasons='|'.join(sorted({choice['reason'] for choice in choices})))
        item.update(configuration_sha256=digest(path),
            selective_exposed_requests=sum(row.get('selective_exposed', False) for row in rows),
            evidence_scope='Marker-free finite-cohort serving. No claim of sustained capacity. Event-based cost controls are separate.')
        items.append(item)
        requests.extend(rows)
        completions.extend(finished)
    destination = ROOT / 'audit'
    destination.mkdir(exist_ok=True)
    for filename, rows in (('serving_results', items), ('request_gate_inventory', requests),
            ('completion_model_audit', completions)):
        write(destination / (filename+'.json'), rows)
        inherited.csv_write(destination / (filename+'.csv'), rows)
    print(json.dumps(dict(serving_cells=len(items), requests=len(requests),
        completed_offloads=len(completions),
        completion_observations=completions[:8])))


if __name__ == '__main__':
    main()
