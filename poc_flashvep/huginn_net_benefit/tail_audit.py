"""Audit extreme delivered ITL without discarding a completed serving start."""
import argparse
from .common import *
from .runtime_observations import records


def summarize(name, limit=192.8212538175284):
    folder = ROOT/'raw'/name
    window = json.loads((folder/'window.json').read_text())
    origin = window['start']
    requests = [row for row in records(folder/'requests.jsonl') if not row.get('warmup')]
    outputs = sorted((event['time'], row['sample_id']) for row in requests
                     for event in row['token_events'])
    samples = list(records(folder/'queue_observations.jsonl'))
    decode = [row for path in folder.glob('atomic_spans_*.json')
              for owner in [json.loads(path.read_text())] if owner['role'] == 'decode'
              for row in owner['time_histogram']]
    values = []
    for row in requests:
        for token_index,event in enumerate(row['token_events']):
            if event['first']:
                continue
            left,right = event['previous_time'], event['time']
            ms = 1000*(right-left)
            values.append(dict(request_index=row['sample_id'], token_event_index=token_index,
                interval_ms=ms, begin=left-origin, end=right-origin))
    values.sort(key=lambda r:r['interval_ms'], reverse=True)
    chosen = [row for index,row in enumerate(values) if index < 10 or row['interval_ms'] > limit]
    for row in chosen:
        left,right = origin+row['begin'], origin+row['end']
        row['other_request_outputs_inside'] = sum(
            left < timestamp < right and request != row['request_index'] for timestamp,request in outputs)
        row['coordinator_sample_times_inside'] = [
            sample['time']-origin for sample in samples if left < sample['time'] < right]
        row['nearest_coordinator_sample_before'] = max(
            (sample['time']-origin for sample in samples if sample['time'] <= left), default=None)
        row['nearest_coordinator_sample_after'] = min(
            (sample['time']-origin for sample in samples if sample['time'] >= right), default=None)
        row['D_submission_count_in_fully_contained_half_second_bins'] = sum(
            entry['submissions'] for entry in decode if left <= entry['time'] and entry['time']+.5 <= right)
    result = dict(name=name, measured_ITL_intervals=len(values), frozen_threshold_ms=limit,
        threshold_exceedances=sum(row['interval_ms'] > limit for row in values),
        extreme_intervals=chosen,
        scope=('First-to-next Host delivery intervals. A coordinator sample gap can have several '
            'causes. These observations do not identify a kernel stall, external interference, '
            'or physical idle. Retain all intervals and the original metrics.'))
    target = ROOT/'audit'/'tail'
    target.mkdir(parents=True, exist_ok=True)
    write(target/(name+'.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    initialize()
    for name in args.names:
        print(json.dumps(summarize(name)))


if __name__ == '__main__':
    main()
