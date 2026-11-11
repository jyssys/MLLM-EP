"""Qualify monotone early KV before any completion or performance claim."""
from .common import *


def main():
    initialize()
    jobs = []
    for cap in (16, 4):
        for arm in ('zero', 'loop'):
            label = f'early9_output{cap}_{arm}'
            for suffix in ('state8', 'state32', 'seed2026110940'):
                source = f'{"pipeline7" if arm == "zero" else "forecast8"}_output{cap}_{arm}_{suffix}'
                cfg = json.loads((ROOT/'configs'/(source+'.json')).read_text())
                cfg.update(name=f'{label}_{suffix}', native_loop_event_transfer=True,
                    early_transfer_depths=[16], phase='IMMUTABLE_EARLY_KV_EXACTNESS_OR_SELECTION')
                if arm == 'loop':
                    cfg.update(monotone_early_KV_protocol=True, progressive_kv=True,
                        require_early_receipt_qualification=True)
                if suffix == 'state32':
                    cfg['required_qualification'] = f'{label}_state8'
                if suffix.startswith('seed'):
                    cfg['required_qualification'] = f'{label}_state32'
                save(cfg)
                jobs.append(cfg['name'])
    write(ROOT/'state/immutable_early_protocol_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
