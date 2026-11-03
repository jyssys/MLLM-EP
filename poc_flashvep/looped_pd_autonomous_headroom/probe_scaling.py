"""Measure larger exact D-local Prefill probes after a joint screen passes."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .verify_long_probes import verify


def main():
    parents = ('graph_joint_validation', 'bulk_followup')
    while time.time() < LONG_START_CUTOFF:
        states = [ROOT/'state'/(name+'.json') for name in parents]
        if all(p.exists() and json.loads(p.read_text())['stage'] in (
                'COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED') for p in states):
            break
        time.sleep(5)
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    ledger = []
    for model in ('ouro', 'huginn'):
        candidates = []
        for parent in parents:
            path = ROOT/'state'/(parent+'_results.json')
            if path.exists():
                candidates.extend(r for r in json.loads(path.read_text())
                                  if r.get('model') == model and r.get('joint_screen_pass'))
        if not candidates:
            ledger.append(dict(model=model, status='NO_QUALIFIED_JOINT_SCREEN'))
            continue
        chosen = candidates[0]
        capture = json.loads((ROOT/'configs'/(chosen['name']+'.json')).read_text())
        original = json.loads((ROOT/'configs'/(capture['matched_clean']+'.json')).read_text())
        for size in (32, 64, 128):
            if time.time() >= LONG_START_CUTOFF:
                break
            cfg = copy.deepcopy(original)
            cfg.update(name=f'probe_scale_{model}_tokens{size}_128', D_probe_tokens=size,
                       phase='D_probe_scaling', timeout=900)
            name = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
            write(ROOT/'state/probe_scaling.json', dict(stage='RUNNING', name=name, time=time.time()))
            row = dict(model=model, name=name, extra_prompt_tokens=size)
            if not execute(name) or not evidence.compare(original['name'], name)['passed']:
                row['status'] = 'BLOCKED_BACKGROUND_PARITY_OR_EXECUTION'
                ledger.append(row)
                break
            checked = verify(name)
            row['correctness'] = checked
            metrics, _, _ = serving.serving_one(cfg)
            row['serving'] = metrics
            row['safe'] = bool(checked.get('exact') and checked.get('full_work') and
                metrics['request_TPOT_ms_p95'] <= .9*metrics['TPOT_threshold_ms'] and
                metrics['token_ITL_ms_p99'] <= metrics['ITL_threshold_ms'])
            row['scope'] = 'Additional independent full-depth Prefill. Not a migrated continuation. This clean run does not inherit GPU activity from another run.'
            ledger.append(row)
            write(ROOT/'state/probe_scaling_results.json', ledger)
            if not row['safe']:
                break
    write(ROOT/'state/probe_scaling_results.json', ledger)
    write(ROOT/'state/probe_scaling.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/probe_scaling.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
