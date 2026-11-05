"""Retest failed startup namespaces with coordinator-preserved warm-up identity."""
import copy
from .common import *
from .prepare import frontier_configuration
from .qualification import qualify


def main():
    initialize()
    target = ROOT / 'state/frontier_repair_v7.json'
    ledger = dict(status='IN_PROGRESS', models={})
    # Ouro startup failures predate the coordinator-origin correction.
    # Huginn reaches these cases later in the main controller. Its immutable
    # v6 receipts determine whether an additional v7 repetition is necessary.
    for model in ('ouro', 'huginn'):
        if model == 'huginn':
            while time.time() < GPU_DEADLINE - 1500:
                gate = json.loads((ROOT / 'state/campaign_v6.json').read_text())
                if gate['status'] == 'QUALIFICATION_FINISHED':
                    break
                time.sleep(5)
            else:
                ledger['models'][model] = dict(status='NOT_RUN_TIME_BUDGET', qualification={})
                write(target, ledger)
                continue
        row = ledger['models'][model] = dict(status='REPAIR_IN_PROGRESS', qualification={})
        plans = ([('token3072', 0, 3072), ('token3840', 0, 3840),
                  ('2d_c2_k2048', 2, 2048), ('2d_c3_k3072', 3, 3072)] if model == 'ouro' else
                 [('token1984', 0, 1984), ('token2016', 0, 2016),
                  ('2d_c28_k1536', 28, 1536), ('2d_c31_k1024', 31, 1024)])
        reference32 = 'b_ouro_native_v5_state32' if model == 'ouro' else 'b_huginn_native_v6_state32'
        for label, c, k in plans:
            if model == 'huginn' and gate['models'][model].get('qualification', {}).get(label, {}).get('passed'):
                continue
            reference_receipt = ROOT / 'raw' / reference32 / 'qualification.json'
            if not reference_receipt.exists() or not json.loads(reference_receipt.read_text()).get('passed'):
                row['qualification'][label] = dict(passed=False, status='BLOCKED_NATIVE_N32')
                continue
            checked = []
            for count, reference in ((8, f'a_{model}_native_v2_state8'), (32, reference32)):
                cfg = frontier_configuration(model, c, k, count,
                    f'b_{model}_{label}_v7_state{count}', True,
                    seed=2026110402 if count == 8 else 2026110403)
                save(cfg)
                checked.append(qualify(cfg['name'], reference))
                if not checked[-1]['passed']:
                    break
            row['qualification'][label] = dict(passed=len(checked) == 2 and all(item['passed'] for item in checked),
                checks=checked, config=cfg['name'])
            write(target, ledger)
        row['status'] = 'REPAIR_FINISHED'
        write(target, ledger)
    ledger['status'] = 'REPAIR_FINISHED'
    write(target, ledger)
    while time.time() < GPU_DEADLINE:
        gate = json.loads((ROOT / 'state/campaign_v6.json').read_text())
        if gate['status'] == 'QUALIFICATION_FINISHED':
            effective = copy.deepcopy(gate)
            effective['source_ledgers'] = ['campaign_v6.json', 'frontier_repair_v7.json']
            for model, row in ledger['models'].items():
                effective['models'][model]['qualification'].update(row['qualification'])
            write(ROOT / 'state/effective_qualification.json', effective)
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
