"""Repeat matched Huginn controls with the actual packed-graph observer."""
from .common import *


def main():
    initialize();jobs=[]
    for arm in ('zero','loop','token','token_group32'):
        for policy in ('original','tuned'):
            source=f'cost_huginn_{arm}_{policy}_serial8_s1'
            cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
            cfg.update(name='core_'+source,observe_core_graph_instance=True,
                serial_output_reference='core_cost_huginn_zero_original_serial8_s1',
                serial_baseline=f'core_cost_huginn_zero_{policy}_serial8_s1',
                phase='INSTRUMENTED_PACKED_RECURRENT_CORE_COMPONENT_CONTROL')
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/core_component_plan.json',dict(jobs=jobs,
        reason='The first observer patched RecurrentGraphs.execute. Actual packed graphs override it with BulkWrittenGraphs.execute. Its recurrent samples were absent, not zero-time. Full submission enclosures remain valid.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
