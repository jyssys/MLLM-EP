"""Retain attempted cells, missing measurements, and memory attribution."""
import csv
from .common import *


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def table(name, rows):
    fields = sorted({key for row in rows for key in row})
    if not fields:
        return
    with (ROOT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def memory_window_summary(values, left, right, physical_total=None):
    selected = [row for row in values if left <= row['time'] <= right]
    if not selected:
        return dict(status='NOT_MEASURED', samples=0)
    totals = {row['device_total_bytes'] for row in selected}
    assert len(totals) == 1
    cuda_total = next(iter(totals))
    total = physical_total or cuda_total
    result = dict(status='MEASURED_SAMPLED_WINDOW', samples=len(selected),
        total_HBM_GiB=total / 1024**3,
        CUDA_usable_total_GiB=cuda_total / 1024**3,
        denominator_source='NVML physical HBM' if physical_total else 'CUDA usable memory fallback',
        scope='Measured-request window only. Sampled maxima can miss short spikes. '
              'The reported allocator peak counter can include earlier startup. '
              'CUDA used memory is not NVML used memory. '
              'Allocator totals overlap weights and KV. Do not stack them.')
    quantities = dict(sampled_CUDA_device_used_peak_bytes=max(cuda_total - row['device_free_bytes'] for row in selected),
        sampled_allocator_allocated_peak_bytes=max(row['allocator_allocated_bytes'] for row in selected),
        sampled_allocator_reserved_peak_bytes=max(row['allocator_reserved_bytes'] for row in selected),
        reported_lifetime_allocator_peak_bytes=max(row['allocator_peak_bytes'] for row in selected),
        sampled_KV_request_allocation_peak_bytes=max(row['KV_used_blocks'] * row['KV_block_bytes'] for row in selected))
    for key, count in quantities.items():
        result[key + '_GiB'] = count / 1024**3
        result[key + '_HBM_percent'] = 100 * count / total
    return result


def main():
    initialize()
    rows, memory, progress_memory, unexecuted = [], [], [], []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = read(path)
        if not isinstance(cfg, dict) or 'name' not in cfg:
            continue
        folder = ROOT / 'raw' / cfg['name']
        receipt = read(folder / 'dispatcher_result.json', {})
        status = ('COMPLETE' if receipt.get('complete') else receipt.get('status', 'FAILED')
                  if receipt else 'UNFINISHED_NAMESPACE' if folder.exists() else 'NOT_RUN')
        row = dict(name=cfg['name'], model=cfg.get('model'), arm=cfg.get('arm'),
            phase=cfg.get('phase'), status=status, count=cfg.get('count'),
            length=cfg.get('length'), output_cap=cfg.get('cap'), depth=cfg.get('depth'),
            chunk=cfg.get('chunk'), P_slots=cfg.get('P_slots'), D_slots=cfg.get('D_slots'),
            token_budget=cfg.get('token_budget'), D_token_budget=cfg.get('D_token_budget'),
            axis=cfg.get('axis'), cut=cfg.get('cut'), prefix=cfg.get('prefix_tokens'),
            nominal_QPS=cfg.get('arrival_rate'), arrival_seed=cfg.get('arrival_seed'),
            independent_start=cfg.get('independent_start'), load=cfg.get('load'),
            GPUs=str(cfg.get('gpus')), dataset=cfg.get('dataset'),
            model_revision=cfg.get('model_revision'), config_SHA256=digest(path),
            dynamic=cfg.get('dynamic', False), heavy_state_probe=cfg.get('state_probe', False),
            verify_transfer=cfg.get('verify_transfer', False),
            event_start_markers=cfg.get('observe_service_timing', False),
            kernel_activity_seconds=cfg.get('kernel_activity_seconds'),
            dispatcher_returncode=receipt.get('returncode'), elapsed_seconds=receipt.get('elapsed_seconds'))
        for filename, prefix in (('new_campaign_source_hashes.json', 'new_source'),
                                 ('parent_campaign_source_hashes.json', 'parent_source')):
            source = folder / filename
            row[prefix + '_manifest_SHA256'] = digest(source) if source.exists() else 'NOT_RECORDED'
        rows.append(row)
        if status != 'COMPLETE':
            unexecuted.append(row)
        totals = {}
        isolation = read(folder / 'isolation.json', {})
        query = isolation.get('GPU_query', '')
        if query:
            for device in csv.DictReader(query.splitlines(), skipinitialspace=True):
                key = next((key for key in device if key.startswith('memory.total')), None)
                if key:
                    totals[int(device['index'])] = float(device[key].split()[0]) * 1024**2
        worker_gpus = {'prefill': (cfg.get('gpus') or [None, None])[0],
                       'decode': (cfg.get('gpus') or [None, None])[-1]}
        for snapshot in folder.glob('memory_inventory_*.json'):
            value = read(snapshot)
            physical = totals.get(worker_gpus.get(value['role']))
            item = dict(name=cfg['name'], model=cfg.get('model'), role=value['role'],
                snapshot=snapshot.name, total_HBM_GiB=physical / 1024**3 if physical else 'NOT_MEASURED',
                scope=value['scope'])
            for key, count in value.items():
                if key.endswith('_bytes'):
                    item[key + '_GiB'] = count / 1024**3
                    item[key + '_HBM_percent'] = 100 * count / physical if physical else 'NOT_MEASURED'
            memory.append(item)
        begin = read(folder / 'measurement_begin.json', {})
        requests = folder / 'requests.jsonl'
        if begin and requests.exists() and receipt.get('complete'):
            measured = [row for line in requests.read_text().splitlines()
                        if not (row := json.loads(line)).get('warmup')]
            if measured:
                right = max(row['finish'] for row in measured)
                for source in folder.glob('memory_progress_*.jsonl'):
                    values = [json.loads(line) for line in source.read_text().splitlines()]
                    if values:
                        progress_memory.append(dict(name=cfg['name'], model=cfg['model'],
                            role=values[0]['role'], source=source.name,
                            **memory_window_summary(values, begin['time'], right,
                                totals.get(worker_gpus.get(values[0]['role'])))))
    table('configuration_inventory.csv', rows)
    table('uncompleted_configuration_inventory.csv', unexecuted)
    table('gpu_memory_inventory_gib_and_percent.csv', memory)
    table('gpu_memory_progress_window_gib_and_percent.csv', progress_memory)
    write(ROOT / 'state/configuration_inventory_summary.json', dict(
        time=time.time(), total_configurations=len(rows),
        complete=sum(row['status'] == 'COMPLETE' for row in rows),
        uncompleted=len(unexecuted),
        scope='Attempt inventory, not a count of clean performance comparisons. Expected failure reproductions, warm-ups, exactness probes, and profiles remain separate.'))


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('configuration_and_memory_inventory'):
        main()
