"""Disjoint finite primitive calibration. Never label this clean serving."""
from .common import *
from .serving import evaluate


def main(effective=False):
    initialize()
    target = ROOT / 'state' / ('primitive_calibration_effective.json' if effective else 'primitive_calibration.json')
    ledger = dict(status='WAITING_N32_QUALIFICATION', models={})
    for model in ('ouro', 'huginn'):
        while time.time() < GPU_DEADLINE - 900:
            path = ROOT / 'state' / ('effective_qualification.json' if effective else 'campaign_v6.json')
            if path.exists():
                gate = json.loads(path.read_text())
                row = gate['models'].get(model, {})
                if row.get('status') in ('EXACT_QUALIFICATION_FINISHED', 'BLOCKED_NATIVE_N32',
                                          'BLOCKED_PRIMARY_NATIVE_CONTINUATION_N8'):
                    break
            time.sleep(5)
        else:
            ledger['models'][model] = dict(status='NOT_RUN_TIME_BUDGET')
            write(target, ledger)
            continue
        choices = row.get('qualification', {})
        results = ledger['models'][model] = dict(status='CALIBRATION_IN_PROGRESS', results=[])
        if not choices.get('native', {}).get('passed'):
            results['status'] = 'BLOCKED_NATIVE_N32'
            write(target, ledger)
            continue
        data_name = f'cost_{model}_disjoint_calibration8.json'
        pool = json.loads((PREVIOUS / 'data' / data_name).read_text())
        destination = ROOT / 'data' / data_name
        if destination.exists():
            assert json.loads(destination.read_text()) == pool
        else:
            write(destination, pool)
        primary = json.loads((ROOT / 'data' / ('confirmation_ouro_L4096.json' if model == 'ouro'
                                              else 'short_confirmation_huginn_L2048.json')).read_text())[:128]
        assert not {row['prompt_id'] for row in primary} & {row['prompt_id'] for row in pool}
        reference = None
        for method, entry in choices.items():
            if not entry.get('passed'):
                continue
            cfg = json.loads((ROOT / 'configs' / (entry['config'] + '.json')).read_text())
            cfg.update(name=f'f_{model}_{method}_serial_primitive8', method=method,
                count=8, state_probe=False, verify_transfer=False, calibration_serial=True,
                phase='ISOLATED_STREAM_CALIBRATION_NOT_CLEAN_SERVING',
                observe_device_execution=True, observe_service_timing=True,
                seed=202611045001, arrival_seed=None, arrival_rate=None,
                arrival_offsets=[0.] * 8, dataset=data_name,
                prompt_ids=[item['prompt_id'] for item in pool], request_caps=[cfg['cap']] * 8,
                timeout=600, startup_timeout=240, independent_start=1, load='serial_calibration',
                data_scope=dict(overlap_with_primary_128=0, records=8,
                    scope='Preserved disjoint calibration records. Closed-loop serial arrivals. Instrumented stream enclosures are not exclusive kernel time.'))
            result = evaluate(cfg, reference)
            results['results'].append(result)
            write(target, ledger)
            if method == 'native':
                if not result['passed']:
                    break
                reference = cfg['name']
        results['status'] = 'PRIMITIVE_CALIBRATION_FINISHED'
        write(target, ledger)
    ledger['status'] = 'PRIMITIVE_CALIBRATION_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
