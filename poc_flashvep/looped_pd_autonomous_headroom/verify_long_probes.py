"""Check each retained long-run probe against native PD with the same seed."""
import copy
from .common import *
from .run import execute
from .campaign import gate


def verify(name):
    folder = ROOT/'raw'/name
    path = folder/'D_probe_requests.jsonl'
    if not path.exists():
        return dict(name=name, status='NO_COMPLETED_PROBE')
    rows = sorted([json.loads(line) for line in path.read_text().splitlines()], key=lambda r: r['seed'])
    if not rows:
        return dict(name=name, status='NO_COMPLETED_PROBE')
    cfg = copy.deepcopy(json.loads((ROOT/'configs'/(name+'.json')).read_text()))
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
        cfg.pop(key, None)
    length = len(rows[0]['input_ids'])
    assert all(len(r['input_ids']) == length for r in rows)
    assert [r['seed'] for r in rows] == list(range(rows[0]['seed'], rows[0]['seed']+len(rows)))
    base_refname = name+'_exact_probe_reference'
    refname = base_refname
    cfg.update(name=refname, length=length, cap=1, count=len(rows),
               state_capacity_tokens=length, seed=rows[0]['seed'],
               request_caps=[1]*len(rows), arrival_offsets=[0.]*len(rows),
               phase='correctness', verify_transfer=True, state_probe=True,
               dataset=refname+'.json', nsys=False, P_only_profiler=False,
               capture_prewarmed=False, capture_until_close=False,
               capture_seconds=0., autonomous_capture_seconds=0., postwarm_capture=False,
               bounded_native_capture_seconds=0.)
    data = [dict(prompt_id=f'probe_{i}', input_ids={cfg['model']:r['input_ids']}) for i,r in enumerate(rows)]
    cfg['prompt_ids'] = [r['prompt_id'] for r in data]
    for version in range(1, 100):
        refname = base_refname if version == 1 else base_refname+f'_v{version}'
        cfg.update(name=refname, dataset=refname+'.json')
        config_path = ROOT/'configs'/(refname+'.json')
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            continue
        previous_run = ROOT/'raw'/refname/'dispatcher_result.json'
        if previous_run.exists() and not json.loads(previous_run.read_text()).get('complete'):
            # Preserve rejected startup and runtime failures. A retry uses a
            # new namespace and must independently pass every check.
            continue
        data_path = ROOT/'data'/cfg['dataset']
        if data_path.exists():
            assert json.loads(data_path.read_text()) == data, 'Reference dataset changed'
        else:
            write(data_path, data)
        if not config_path.exists():
            write(config_path, cfg)
        break
    else:
        raise RuntimeError('No unused exact-reference namespace')
    okay = execute(refname)
    if okay:
        ref = [json.loads(line) for line in (ROOT/'raw'/refname/'requests.jsonl').read_text().splitlines()]
        ref = {r['sample_id']: r for r in ref if not r.get('warmup')}
        okay = gate(refname) and all(ref[i]['seed'] == r['seed'] and
            ref[i]['input_ids'] == r['input_ids'] and ref[i]['output_ids'] == r['output_ids'] and
            ref[i]['exit_depths'] == r['exit_depths'] for i,r in enumerate(rows))
    before = json.loads((folder/'before_measured.json').read_text())
    after = json.loads((folder/'after_measured.json').read_text())
    requests = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    requests = [r for r in requests if not r.get('warmup')]
    expected = sum((r['input_tokens']+r['output_tokens']-1)*cfg['depth'] for r in requests)
    expected += sum(len(r['input_ids'])*cfg['depth'] for r in rows)
    observed = sum(v['issued_loop_rows']-before[k]['issued_loop_rows'] for k,v in after.items())
    result = dict(name=name, reference=refname, exact=okay, completed_probes=len(rows),
                  expected_recurrent_rows=expected, observed_recurrent_rows=observed,
                  full_work=observed == expected,
                  scope='Correctness evidence. This does not establish P saturation or a serving speedup.')
    previous = folder/'long_probe_correctness.json'
    if not previous.exists():
        write(previous, result)
    elif json.loads(previous.read_text()) != result:
        write(folder/(refname+'_correctness.json'), result)
    return result


def main():
    while time.time() < LONG_START_CUTOFF:
        path = ROOT/'state/long_validation.json'
        if path.exists() and json.loads(path.read_text())['stage'] in ('LONG_VALIDATION_COMPLETE', 'FAILED_REVIEW_REQUIRED'):
            break
        time.sleep(5)
    results = []
    for folder in sorted((ROOT/'raw').glob('long_deep_*_probe16_128')):
        if time.time() >= LONG_START_CUTOFF:
            break
        status = folder/'status.json'
        if not status.exists():
            continue
        results.append(verify(folder.name))
        write(ROOT/'state/long_probe_verification_results.json', results)


if __name__ == '__main__':
    main()
