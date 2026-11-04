"""Refresh CPU evidence after every GPU controller reaches a terminal state."""
import importlib
from .common import *


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    stages = []
    with quiet_analysis('terminal_CPU_evidence_and_figures'):
        for name in ('static_campaign', 'dynamic_campaign'):
            state = json.loads((ROOT / 'state' / (name + '.json')).read_text())
            assert state['status'].endswith('_CAMPAIGN_FINISHED'), 'Do not analyze an active GPU campaign'
        for name in ('refresh_analysis', 'online_decision_analysis', 'marker_free_analysis',
                     'transport_timings', 'dataset_inventory', 'inventory', 'activity_figures', 'figures'):
            begin = time.time()
            print('CPU_ANALYSIS_START', name, flush=True)
            importlib.import_module('.' + name, __package__).main()
            stages.append(dict(module=name, start=begin, finish=time.time()))
            write(ROOT / 'state/closeout_analysis_receipt.json', dict(status='IN_PROGRESS', stages=stages))
            print('CPU_ANALYSIS_END', name, flush=True)
        write(ROOT / 'state/closeout_analysis_receipt.json', dict(status='COMPLETE', stages=stages,
            scope='CPU derived evidence and figures only. No GPU experiments or acceptance thresholds changed.'))


if __name__ == '__main__':
    main()
