"""Repair Local stage geometry selection without changing Serving runtime."""
import json
from .common import *


def main():
    initialize();parents=(
        'reserve_capacity_huginn_L3072_B12_start1',
        'reserve_capacity_huginn_L3900_B12_start1',
        'capacity_ouro_L8192_O16_B124_start1',
        'capacity_ouro_L16384_O16_B124_start1',
        'extended_capacity_ouro_L18432_O16_B12_start1')
    mapping={};jobs=[]
    for parent in parents:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name='matched_graph_'+parent
        cfg.update(name=name,capacity_native_prefill_geometry=True,
            phase='MATCHED_NATIVE_PREFILL_GRAPH_CAPACITY_RECHECK',
            selection_reason='The Local stage label selected no primary graph geometries. Map only this capacity role to the Native P geometry set. Model math, KV bookkeeping, batch limits, and Serving configs remain unchanged.')
        save(cfg);jobs.append(name);mapping[parent]=name
    write(ROOT/'state/matched_graph_capacity_recheck_plan.json',dict(jobs=jobs,replacements=mapping,
        scope='Measurement repair only. The prior fallback-path cohorts remain available and cannot establish proximity to the strongest matched Native graph reference. Prepared D still runs on the same isolated local GPU and is not online D capacity.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
