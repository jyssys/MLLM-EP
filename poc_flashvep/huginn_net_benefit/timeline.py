"""Keep GPU activity unknown when the serving evidence contains Host observations."""
import argparse
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', action='append', required=True, help='LABEL=COMPLETED_N128_CONFIG')
    parser.add_argument('--tag', default='selection')
    args = parser.parse_args()
    initialize()
    from ..completion_aware_loop_pd.timeline import run_data, HTML
    values = []
    sources = []
    arrival_seeds = set()
    for item in args.arm:
        label, name = item.split('=', 1)
        cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
        assert cfg['model'] == 'huginn' and cfg['count'] == 128
        assert analysis().metrics(name)['complete']
        arrival_seeds.add(cfg['arrival_seed'])
        value = run_data(ROOT, name, label)
        folder = ROOT/'raw'/name
        begin = json.loads((folder/'measurement_begin.json').read_text())['time']
        native = {row['request_id']:row for row in json.loads(
            (folder/'stats.json').read_text())['metrics']}
        prompt = [row for path in folder.glob('atomic_spans_*.json')
            for owner in [json.loads(path.read_text())] if owner['role'] == 'decode'
            for row in owner['prompt_spans'] if row['time'] >= begin]
        for request in value['requests']:
            matched = native[f'r0_q{request["index"]}']
            admitted = matched['timings'].get('admitted')
            request['admitted'] = (matched['created_monotonic']+admitted
                                   if admitted is not None else None)
            spans = [row for row in prompt if row['seed'] == request['seed']]
            rows = sum(row['count']*row['loop_count'] for row in spans)
            request['actual_D_recurrent_rows'] = rows
            if cfg.get('intervention_method', 'native') == 'native' and spans:
                frontier = cfg.get('native_frontier')
                if frontier and frontier[0] == 0:
                    request['offloaded_suffix_tokens'] = cfg['length']-frontier[1]
                elif all(row['count'] == cfg['length'] for row in spans):
                    request['offloaded_loops'] = rows//cfg['length']
                if not request['d']:
                    start = min(row['time'] for row in spans)
                    request['d'] = [dict(start=start, end=request['first'],
                        label='D submission → first output, includes waits',
                        kind='Observed Host completion enclosure, not kernel duration')]
        value['configuration_sha256'] = digest(ROOT/'configs'/(name+'.json'))
        value['physical_GPU_mapping'] = cfg['gpus']
        value['arrival_seed'] = cfg['arrival_seed']
        moved = [r for r in value['requests'] if r['actual_D_recurrent_rows']]
        value['later_P_work_observed'] = dict(
            moved_requests=len(moved),
            later_P_submission_during_D_receipt_to_first_output=sum(
                any(request['received'] <= span['start'] < request['first']
                    for other in value['requests'] if other['index'] > request['index']
                    for span in other['p'])
                for request in moved if request['received'] is not None),
            scope='Paired Host milestones only. A source submission during this interval does not prove physical GPU compute overlap.')
        values.append(value)
        sources.append(name)
    assert len(arrival_seeds) == 1, 'A comparison must use one paired Arrival seed'
    folder = ROOT/'figures'
    folder.mkdir(exist_ok=True)
    data = {'huginn': values}
    name = args.tag+'_observed_timeline'
    write(folder/(name+'_data.json'), data)
    html = HTML.replace('<option>ouro</option><option>huginn</option>', '<option>huginn</option>')
    html = html.replace('Native / old / new', 'Huginn Native / controls / candidates')
    html = html.replace('Orange: P submission-to-ready host enclosure.',
        'Gray: request arrival to coordinator admission. Orange: P submission-to-ready host enclosure.')
    html = html.replace('D submission-to-observed-completion host enclosure.',
        'D submission-to-observed-completion host enclosure. Static continuation enclosures can include waiting and first-output work.')
    html = html.replace('for(let b of r.p)bar',
        "if(r.admitted!==null)bar(r.arrival,r.admitted,top+15,'#788598','Queue to admission');\nfor(let b of r.p)bar")
    html = html.replace('__DATA__', json.dumps(data))
    (folder/(name+'.html')).write_text(html, encoding='utf-8')
    path = ROOT/'figure_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else []
    write(path, [row for row in manifest if row['name'] != name]+[dict(
        name=name, files=[f'figures/{name}.html', f'figures/{name}_data.json'],
        sources=sources,
        scope='Paired actual N128 requests. Host submission, completion observations, transfer intervals, and output deliveries only. No GPU kernel bars or physical idle windows are inferred.')])
    print(json.dumps(dict(html=str(folder/(name+'.html')),
        host_progress=[dict(name=value['name'], **value['later_P_work_observed']) for value in values])))


if __name__ == '__main__':
    main()
