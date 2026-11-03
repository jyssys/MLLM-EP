"""Check exact D warm-up state against native PD, then run clean probe controls."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .campaign import gate
from .verify_long_probes import verify
from .analysis_lane import quiet_analysis


def warm_reference(model, source, reference_name=None):
    folder = ROOT/'raw'/source
    warm = json.loads((folder/'D_probe_warmup_finished.json').read_text())
    states = [r for path in folder.glob('prompt_states_*.jsonl')
              for r in map(json.loads, path.read_text().splitlines())
              if r.get('prompt_tokens') == len(warm['input_ids'])]
    assert len(states) == 1, 'Expected one measured warm-up state'
    candidate = states[0]
    cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s', 'D_probe_warmup_tokens'):
        cfg.pop(key, None)
    name = reference_name or f'd_graph_{model}_warm_native_state_reference'
    cfg.update(name=name, D_prefill_graph_rows=0, length=len(warm['input_ids']),
        cap=1, count=1, seed=candidate['seed'], request_caps=[1], arrival_offsets=[0.],
        state_capacity_tokens=len(warm['input_ids']), dataset=name+'.json',
        prompt_ids=['exact_warm_state'], phase='correctness', verify_transfer=True, state_probe=True)
    for path, value in ((ROOT/'data'/cfg['dataset'], [dict(prompt_id='exact_warm_state', input_ids={model:warm['input_ids']})]),
                        (ROOT/'configs'/(name+'.json'), cfg)):
        if path.exists():
            assert json.loads(path.read_text()) == value, 'Preserve existing reference configuration'
        else:
            write(path, value)
    result = dict(source=source, reference=name, complete=execute(name))
    if not result['complete']:
        return result
    baseline = [r for path in (ROOT/'raw'/name).glob('prompt_states_*.jsonl')
                for r in map(json.loads, path.read_text().splitlines())
                if r.get('seed') == candidate['seed'] and r.get('prompt_tokens') == cfg['length']]
    requests = [r for r in map(json.loads, (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]
    result.update(state_exact=bool(baseline) and all(
        all(r[k] == candidate[k] for k in ('hidden_sha256', 'planes')) for r in baseline),
        output_exact=len(requests) == 1 and requests[0]['output_ids'] == warm['output_ids']
            and requests[0]['exit_depths'] == warm['exit_depths'], native_gate=gate(name),
        compared_native_states=len(baseline), compared_KV_planes=len(candidate['planes']))
    write(folder/'D_graph_warm_state_correctness.json', result)
    return result


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    ledger = []
    for model in ('huginn', 'ouro'):
        source = f'd_probe_graph_{model}_gate16'
        result = ROOT/'raw'/source/'dispatcher_result.json'
        while not result.exists() and time.time() < LONG_START_CUTOFF:
            time.sleep(5)
        if not result.exists() or not json.loads(result.read_text()).get('complete'):
            ledger.append(dict(model=model, status='NO_COMPLETE_GATE'))
            continue
        if not gate(source):
            ledger.append(dict(model=model, status='BLOCKED_TRANSFER_OR_WORK'))
            continue
        checked = warm_reference(model, source)
        row = dict(model=model, state_correctness=checked)
        if not all(checked.get(k) for k in ('complete', 'state_exact', 'output_exact', 'native_gate')):
            row['status'] = 'BLOCKED_D_GRAPH_STATE'
            ledger.append(row)
            continue
        original = f'bulk_joint_{model}_start1_probe16_128'
        cfg = copy.deepcopy(json.loads((ROOT/'configs'/(original+'.json')).read_text()))
        cfg.update(name=f'd_graph_{model}_state_verified_matched128', D_prefill_graph_rows=16,
            phase='D_local_prefill_graph_control')
        name = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/d_graph_state_followup.json', dict(stage='MATCHED_128', name=name, time=time.time()))
        row['name'] = name
        row['complete'] = execute(name)
        if row['complete']:
            row['background_parity'] = evidence.compare(original, name)
            row['probe_correctness'] = verify(name)
            with quiet_analysis(name+'_metrics'):
                row['serving'], _, _ = serving.serving_one(cfg)
        row['scope'] = 'Native full-depth D-local Prefill. Not migrated continuation, and not same-run P saturation proof.'
        ledger.append(row)
        write(ROOT/'state/d_graph_state_followup_results.json', ledger)
    write(ROOT/'state/d_graph_state_followup_results.json', ledger)
    write(ROOT/'state/d_graph_state_followup.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/d_graph_state_followup.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
