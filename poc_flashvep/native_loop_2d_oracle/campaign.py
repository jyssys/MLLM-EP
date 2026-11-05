"""Gate all paths at N8 and N32 before a bounded matched serving screen."""
import traceback
from .common import *
from .prepare import configuration, frontier_configuration
from .qualification import qualify
from .run import execute


def main():
    initialize()
    ledger = dict(status='EXACT_QUALIFICATION_IN_PROGRESS', models={})
    target = ROOT / 'state/campaign_v6.json'
    def persist():
        write(target, ledger)
    for model in ('ouro', 'huginn'):
        row = ledger['models'][model] = dict(qualification={}, serving=[])
        reference8 = f'a_{model}_native_v2_state8'
        primary8 = f'a_{model}_partial_v5_state8' if model == 'ouro' else f'a_{model}_partial_v6_state8'
        if model == 'huginn':
            save(configuration(model, 'partial', 8, primary8, True, cut=31))
        primary = qualify(primary8, reference8)
        row['primary8'] = primary
        persist()
        if not primary['passed']:
            row['status'] = 'BLOCKED_PRIMARY_NATIVE_CONTINUATION_N8'
            persist()
            continue
        cuts = [3, 2] if model == 'ouro' else [31, 28]
        plans = [('native', None), ('zero', None)] + [(f'loop{cut}', cut) for cut in cuts]
        for label, cut in plans:
            path = 'native' if label == 'native' else 'zero' if label == 'zero' else 'partial'
            checked = []
            if label == 'zero':
                cfg = configuration(model, path, 8, f'b_{model}_{label}_v6_state8', True)
                if model == 'ouro':
                    cfg = json.loads((ROOT / 'configs/b_ouro_zero_v5_state8.json').read_text())
                else:
                    save(cfg)
                checked.append(qualify(cfg['name'], reference8))
                if not checked[-1]['passed']:
                    row['qualification'][label] = dict(passed=False, checks=checked)
                    persist()
                    continue
            if cut is not None and cut != cuts[0]:
                cfg = configuration(model, path, 8, f'b_{model}_{label}_v6_state8', True, cut=cut)
                save(cfg)
                checked.append(qualify(cfg['name'], reference8))
                if not checked[-1]['passed']:
                    row['qualification'][label] = dict(passed=False, checks=checked)
                    persist()
                    continue
            name = f'b_{model}_{label}_v6_state32'
            if model == 'ouro' and label == 'native':
                name = 'b_ouro_native_v5_state32'
            cfg = configuration(model, path, 32, name, True, cut=cut, seed=2026110403)
            if model == 'ouro' and label == 'native':
                cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
            else:
                save(cfg)
            reference32 = 'b_ouro_native_v5_state32' if model == 'ouro' else 'b_huginn_native_v6_state32'
            checked.append(qualify(name, None if label == 'native' else reference32))
            row['qualification'][label] = dict(passed=all(item['passed'] for item in checked),
                                              checks=checked, config=name)
            persist()
            if label == 'native' and not checked[-1]['passed']:
                break
        if not row['qualification'].get('native', {}).get('passed'):
            row['status'] = 'BLOCKED_NATIVE_N32'
            persist()
            continue
        choices = ([('token3072', 0, 3072), ('token3840', 0, 3840),
                    ('2d_c2_k2048', 2, 2048), ('2d_c3_k3072', 3, 3072)] if model == 'ouro' else
                   [('token1984', 0, 1984), ('token2016', 0, 2016),
                    ('2d_c28_k1536', 28, 1536), ('2d_c31_k1024', 31, 1024)])
        for label, c, k in choices:
            checked = []
            for count, reference in ((8, reference8), (32, reference32)):
                cfg = frontier_configuration(model, c, k, count,
                    f'b_{model}_{label}_v6_state{count}', True, seed=2026110402 if count == 8 else 2026110403)
                save(cfg)
                checked.append(qualify(cfg['name'], reference))
                if not checked[-1]['passed']:
                    break
            row['qualification'][label] = dict(passed=len(checked) == 2 and all(item['passed'] for item in checked),
                                              checks=checked, config=cfg['name'])
            persist()
        row['status'] = 'EXACT_QUALIFICATION_FINISHED'
        persist()
    ledger['status'] = 'QUALIFICATION_FINISHED'
    persist()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        initialize()
        write(ROOT / 'state/campaign_exception.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
