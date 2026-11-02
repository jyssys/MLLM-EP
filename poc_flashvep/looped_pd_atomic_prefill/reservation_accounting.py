"""Count native D reservations separately from requests that can Decode."""
from .common import *
from .evidence import status


def clipped_seconds(interval, left, right):
    return max(0., min(interval[1], right) - max(interval[0], left))


def occupancy(intervals, left, right):
    events = [(left, 0), (right, 0)]
    for start, end in intervals:
        start, end = max(left, start), min(right, end)
        if end > start:
            events.extend(((start, 1), (end, -1)))
    events.sort()
    levels = []
    count, previous, total, peak = 0, left, 0., 0
    for current, change in events:
        if current > previous:
            levels.append((count, current - previous))
            total += count * (current - previous)
        count += change
        peak = max(peak, count)
        previous = current
    return dict(mean=total / (right-left), peak=peak, request_seconds=total,
        zero_fraction=sum(duration for count,duration in levels if count == 0)/(right-left))


def analyze():
    from .analyze import save_csv
    output = []
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if not isinstance(cfg, dict) or cfg.get('arm') != 'PD_G1':continue
        if cfg.get('phase') not in ('qps_pilot', 'final_repeat'):continue
        name = cfg.get('name')
        if not name or status(name) != 'COMPLETE':continue
        folder = ROOT/'raw'/name
        if not (folder/'closed_stats.json').exists() or not (folder/'window.json').exists():continue
        window = json.loads((folder/'window.json').read_text())
        left = window['start'] + .2*(window['last_arrival']-window['start'])
        right = window['start'] + .9*(window['last_arrival']-window['start'])
        if right <= left:continue
        intervals = dict(reserved=[], pending=[], active=[])
        incomplete = []
        for row in json.loads((folder/'closed_stats.json').read_text())['metrics']:
            if not row['request_id'].startswith('r0_'):continue
            times = row['timings'];origin = row['created_monotonic']
            if not all(key in times for key in ('d_reserved','d_activated','finished')):
                incomplete.append(row['request_id']);continue
            reserved, activated, finished = (origin+times[k] for k in ('d_reserved','d_activated','finished'))
            if not reserved <= activated <= finished:
                incomplete.append(row['request_id']);continue
            intervals['reserved'].append((reserved, finished))
            intervals['pending'].append((reserved, activated))
            intervals['active'].append((activated, finished))
        if not intervals['reserved']:continue
        values = {key:occupancy(items,left,right) for key,items in intervals.items()}
        denominator = values['reserved']['request_seconds']
        conservation_error = values['reserved']['request_seconds']-values['pending']['request_seconds']-values['active']['request_seconds']
        result = dict(name=name,model=cfg['model'],length=cfg['length'],output_cap=cfg['cap'],chunk=cfg['chunk'],
            phase=cfg['phase'],central_window_s=right-left,measured_requests=len(intervals['reserved']),
            incomplete_metric_requests=incomplete,
            reserved_D_contexts_mean=values['reserved']['mean'],reserved_D_contexts_peak=values['reserved']['peak'],
            pending_D_contexts_mean=values['pending']['mean'],pending_D_contexts_peak=values['pending']['peak'],
            active_D_contexts_mean=values['active']['mean'],active_D_contexts_peak=values['active']['peak'],
            no_active_D_request_fraction=values['active']['zero_fraction'],
            pending_fraction_of_reserved_context_time=values['pending']['request_seconds']/denominator if denominator else None,
            active_fraction_of_reserved_context_time=values['active']['request_seconds']/denominator if denominator else None,
            context_time_conservation_error_s=conservation_error,
            D_KV_blocks=cfg.get('D_kv_blocks',cfg.get('kv_blocks')),
            max_draining_requests=cfg.get('native_transport_limits',{}).get('max_draining_requests',8),
            evidence='MEASURED_HOST_NATIVE_RESERVATION_INTERVALS',
            scope='D activation is a coordinator boundary. It is not a CUDA kernel interval or a count of simultaneous physical GEMM rows. A missing active request is not proof of GPU idle.')
        assert abs(conservation_error) < 1.e-5, result
        output.append(result)
    save_csv('native_D_reservation_accounting.csv',output)
    write(ROOT/'native_D_reservation_accounting.json',output)
    return output


if __name__ == '__main__':analyze()
