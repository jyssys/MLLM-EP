"""Check the smallest measured advantage without per-loop inner events."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for arm in ('loop','token'):
        original=f'priority_width_huginn_S512_{arm}_grouped_tuned_serial8_s1'
        cfg=json.loads((ROOT/'configs'/(original+'.json')).read_text())
        cfg.update(name=original.replace('_serial8_s1','_outeronly_serial8_s2'),
            priority_inner_core_events=False,observe_core_graph_instance=False,
            phase='SERIAL_DIFFERENTIAL_CORE_EVENT_OVERHEAD_CONTROL',
            outeronly_reference=original,
            P_saving_scope='The marked Native baseline has different inner observers. Do not use this cell for a paired P-saving claim.')
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/priority_outeronly_control_plan.json',dict(jobs=jobs,
        scope='Two outer CUDA Events remain on each single D submission. Per-core event counts no longer differ between arms. Separate process starts, not marker-free online Serving.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
