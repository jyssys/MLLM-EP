"""Refresh derived fields without changing configs, tokens, or device traces."""
from .common import *


def main():
    initialize()
    from .metrics import summarize
    from .work_accounting import summarize as accounting
    from .lifetime_gate import check as lifetime
    from ..looped_pd_atomic_prefill import evidence
    rows = []
    for folder in sorted((ROOT / 'raw').iterdir()):
        receipt = folder / 'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            continue
        cfg = json.loads((ROOT / 'configs' / (folder.name + '.json')).read_text())
        if (cfg.get('state_probe') or cfg.get('verify_transfer') or cfg.get('calibration_serial')
                or not cfg.get('observe_service_timing') or cfg.get('worker') != 'rlt_worker'):
            continue
        row = dict(name=folder.name, model=cfg['model'], passed=False)
        try:
            row['metrics'] = summarize(folder.name)
            row['accounting'] = accounting(folder.name)
            row['work'] = evidence.full_work(folder.name)
            row['lifetime'] = lifetime(folder.name)
            analysis = folder / 'profile_analysis.json'
            if analysis.exists():
                reference = json.loads(analysis.read_text())['clean']
            elif folder.name.startswith('l_'):
                reference = f'l_{cfg["model"]}_{cfg["load"]}_native_start{cfg["independent_start"]}_{cfg["count"]}'
            else:
                reference = f'f_{cfg["model"]}_{cfg["load"]}_native_start1_128'
            row['output'] = evidence.compare(reference, folder.name)
            row['passed'] = all(row[key]['passed'] for key in ('output', 'work', 'lifetime'))
        except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
            row['status'] = 'DERIVED_ANALYSIS_REQUIRES_REVIEW'
            row['error'] = repr(error)
        rows.append(row)
    write(ROOT / 'state/refreshed_analysis.json', dict(time=time.time(), results=rows,
        producer_SHA256=digest(SOURCE / 'metrics.py'), work_producer_SHA256=digest(SOURCE / 'work_accounting.py'),
        scope='Derived summaries only. Original run configurations, token IDs, CUDA event records, kernel records, and failed logs remain unchanged.'))


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('final_derived_analysis_refresh'):
        main()
