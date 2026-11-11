"""Summarize stock AR diagnostics separately from the Huginn campaign."""
from .ar_diagnostic import complete
from .common import *
from .cost_analysis import quantiles
from .runtime_observations import records


def summarize(name):
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    folder = ROOT/'raw'/name
    if not folder.exists():
        return dict(name=name, status='NOT_RUN')
    if not complete(folder, cfg):
        return dict(name=name, status='INCOMPLETE')
    window = json.loads((folder/'window.json').read_text())
    begin, last, end = [window[k] for k in ('start', 'last_arrival', 'end')]
    a, b = begin+.2*(last-begin), begin+.9*(last-begin)
    requests = list(records(folder/'requests.jsonl'))
    intervals, grouped_deliveries = [], 0
    for request in requests:
        events = request['token_events']
        for previous, current in zip(events, events[1:]):
            if current['count']-previous['count'] == 1:
                intervals.append(1000*(current['time']-previous['time']))
            else:
                grouped_deliveries += 1
    native = {}
    for path in folder.glob('native_*.json'):
        item = json.loads(path.read_text())
        if 'queue_samples' not in item:
            continue
        samples = [r for r in item['queue_samples'] if a <= r['time'] <= b]
        if not samples:
            continue
        native[item['role']] = dict(samples=len(samples),
            nonempty_wait_queue_samples=sum(r['waiting']>0 for r in samples),
            running=quantiles([r['running'] for r in samples]),
            waiting=quantiles([r['waiting'] for r in samples]),
            KV_utilization=quantiles([r['KV_usage'] for r in samples]),
            KV_allocation_refusal_events=sum(begin <= r['time'] <= end
                for r in item['KV_allocation_refusals']))
    return dict(name=name, status='COMPLETE', requested=cfg['count'], completed=len(requests),
        configured_QPS=cfg['arrival_rate'], realized_arrival_QPS=(len(requests)-1)/(last-begin),
        completed_QPS=len(requests)/(end-begin),
        input_tokens_s=sum(r['input_tokens'] for r in requests)/(end-begin),
        output_tokens_s=sum(r['output_tokens'] for r in requests)/(end-begin),
        TTFT_ms=quantiles([r['ttft_ms'] for r in requests]),
        TPOT_ms=quantiles([r['tpot_ms'] for r in requests]),
        E2E_ms=quantiles([r['e2e_ms'] for r in requests]), ITL_ms=quantiles(intervals),
        grouped_delivery_events=grouped_deliveries, central_stage_samples=native,
        physical_GPU_kernel_busy='NOT_MEASURED', physical_D_idle='NOT_MEASURED',
        scope=('Stock AR diagnostic. Finite-cohort drain rate. ITL uses single-token '
            'delivery intervals only. Stage samples describe scheduler observations. '
            'They do not establish concurrent physical P saturation and Decode-safe D slack.'))


def main():
    initialize()
    jobs = json.loads((ROOT/'state/ar_diagnostic_jobs.json').read_text())
    rows = [summarize(name) for name in jobs]
    write(ROOT/'audit/ar_diagnostic_summary.json', rows)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
