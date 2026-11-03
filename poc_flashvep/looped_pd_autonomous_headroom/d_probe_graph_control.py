"""Check native graph replay for exact D-local Prefill, before larger probes."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .campaign import gate
from .verify_long_probes import verify
from .analysis_lane import quiet_analysis


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence, analyze as serving
    ledger = []
    for model in ('huginn', 'ouro'):
        source = f'bulk_joint_{model}_start1_probe16_128'
        result = ROOT/'raw'/source/'dispatcher_result.json'
        while not result.exists() and time.time() < LONG_START_CUTOFF:
            time.sleep(5)
        if not result.exists() or not json.loads(result.read_text()).get('complete'):
            continue
        cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
        cfg.update(name=f'd_probe_graph_{model}_gate16', D_prefill_graph_rows=16,
            phase='correctness', verify_transfer=True, state_probe=True,
            arrival_seed=2026110701, timeout=900)
        name = put(cfg, 16, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/d_probe_graph_control.json', dict(stage='CORRECTNESS', name=name, time=time.time()))
        row = dict(model=model, gate_name=name)
        if time.time() >= LONG_START_CUTOFF or not execute(name) or not gate(name):
            row['status'] = 'BLOCKED_CORRECTNESS'
            ledger.append(row)
            continue
        checked = verify(name)
        row['gate_correctness'] = checked
        if not checked.get('exact') or not checked.get('full_work'):
            row['status'] = 'BLOCKED_PROBE_CORRECTNESS'
            ledger.append(row)
            continue
        cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
        cfg.update(name=f'd_probe_graph_{model}_matched128', D_prefill_graph_rows=16,
            phase='D_local_prefill_graph_control')
        name = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
        write(ROOT/'state/d_probe_graph_control.json', dict(stage='MATCHED_128', name=name, time=time.time()))
        if not execute(name) or not evidence.compare(source, name)['passed']:
            row['status'] = 'BLOCKED_BACKGROUND_PARITY_OR_EXECUTION'
        else:
            row['correctness'] = verify(name)
            with quiet_analysis(name+'_metrics'):
                row['serving'], _, _ = serving.serving_one(cfg)
            row['name'] = name
            row['status'] = 'MEASURED_REVIEW_REQUIRED'
            row['scope'] = 'Generic native graph optimization for D-local Prefill. No migrated request, work reduction, or joint P saturation claim.'
        ledger.append(row)
        write(ROOT/'state/d_probe_graph_control_results.json', ledger)
    write(ROOT/'state/d_probe_graph_control_results.json', ledger)
    write(ROOT/'state/d_probe_graph_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/d_probe_graph_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
