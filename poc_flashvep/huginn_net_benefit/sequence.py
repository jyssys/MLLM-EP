"""Run paired cells serially and restore approved GPU work on exit."""
import argparse
import subprocess
import traceback
from .common import *


def observe_completed(name, cfg, item):
    item['metrics'] = analysis().metrics(name)
    if not cfg.get('state_probe'):
        return
    gate = analysis().qualify_frozen_reference(name)
    if cfg.get('post_decode_state'):
        from .qualification import final_decode
        gate['final_decode'] = final_decode(name, cfg.get('post_decode_reference'))
        gate['passed'] &= gate['final_decode']['passed']
    if cfg.get('require_early_receipt_qualification'):
        from .monotone_early_protocol import qualification_receipt
        gate['immutable_early_receipt'] = qualification_receipt(ROOT/'raw'/name)
        gate['passed'] &= gate['immutable_early_receipt']['passed']
    if cfg.get('require_effective_quantum_qualification'):
        from .quantum_audit import audit
        gate['effective_quantum'] = audit(name)
        gate['passed'] &= gate['effective_quantum']['passed']
    gate['campaign_gate_passed'] = (gate['passed'] and
        (gate.get('actual_dynamic_continuation_tested', False)
         if cfg.get('require_actual_offload_qualification') else True))
    item['qualification'] = gate
    write(ROOT/'raw'/name/'qualification_with_frozen_reference.json', gate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--independent', action='store_true')
    args = p.parse_args()
    names = args.names
    initialize()
    history = ROOT / 'state' / f'sequence_{os.getpid()}_{time.time_ns()}.json'
    results = []
    write(history, dict(pid=os.getpid(), status='RUNNING', names=names, results=results, time=time.time()))
    write(ROOT / 'state/controller.json', dict(pid=os.getpid(), status='RUNNING', names=names, time=time.time()))
    try:
        for name in names:
            cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
            if time.time() >= LONG_START_CUTOFF:
                results.append(dict(name=name, status='NOT_RUN_BUDGET_CUTOFF', time=time.time()))
                continue
            result = subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python', '-m',
                'poc_flashvep.huginn_net_benefit.run', name], cwd=REPO)
            item = dict(name=name, returncode=result.returncode, time=time.time())
            if result.returncode == 0:
                try:
                    observe_completed(name, cfg, item)
                except Exception:
                    error = dict(status='OBSERVATION_FAILED', traceback=traceback.format_exc(), time=time.time())
                    item['observation_error'] = error
                    write(ROOT/'raw'/name/'campaign_observation_error.json', error)
                    if cfg.get('state_probe'):
                        item['qualification'] = dict(passed=False, campaign_gate_passed=False, error=error)
                        write(ROOT/'raw'/name/'qualification_with_frozen_reference.json', item['qualification'])
            results.append(item)
            write(history, dict(pid=os.getpid(), status='RUNNING', names=names, results=results, time=time.time()))
            write(ROOT / 'state/sequence_latest.json', results)
            m = item.get('metrics', {})
            print(json.dumps(dict(name=name, returncode=result.returncode,
                TTFT_p95_ms=m.get('TTFT_p95_ms'), ITL_p99_ms=m.get('ITL_p99_ms'),
                completed_QPS=m.get('completed_QPS'), offloaded=m.get('offloaded_requests'),
                counts=m.get('accepted_loop_counts'),
                qualification=item.get('qualification', {}).get('campaign_gate_passed'))), flush=True)
            if (result.returncode or item.get('observation_error') or
                    (item.get('qualification') and not item['qualification']['campaign_gate_passed'])):
                if not args.independent:
                    break
    finally:
        configure()
        from ..looped_pd_autonomous_headroom.common import resources
        owner = resources()
        if not owner.processes():
            owner.restore(list(range(8)))
        write(ROOT / 'state/controller.json', dict(pid=os.getpid(), status='CLOSED', results=results, time=time.time()))
        write(history, dict(pid=os.getpid(), status='CLOSED', names=names, results=results, time=time.time()))


if __name__ == '__main__':
    main()
