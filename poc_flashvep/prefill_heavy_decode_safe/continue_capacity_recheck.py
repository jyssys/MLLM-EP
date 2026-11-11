"""Require actual graph replays before publishing a corrected P reference."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .capacity_analysis import analyze


def main():
    initialize()
    while not (ROOT/'state/huginn_context_edge_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    plan=json.loads((ROOT/'state/matched_graph_capacity_recheck_plan.json').read_text());out=[];mapping={}
    for old,name in plan['replacements'].items():
        if not sequence([name],'matched_capacity_'+name):continue
        counts=[json.loads(s) for s in (ROOT/'raw'/name/'native_prefill_graph_counts.jsonl').read_text().splitlines()]
        assert sum(r['replays'] for r in counts)>0,'Reference still bypasses the Native Prefill graphs'
        value=analyze(name);assert value['strongest_finite_P_reference']
        mapping[old]=name;out.append(value)
        write(ROOT/'state/matched_graph_capacity_reference_map.json',dict(replacements=mapping,time=time.time(),
            scope='Only successful finite cohorts with actual Native Prefill graph replays enter this map. No old cohort or Serving configuration changes.'))
    write(ROOT/'analysis/matched_graph_capacity_references.json',out)
    write(ROOT/'state/matched_graph_capacity_recheck_complete.json',dict(complete=True,time=time.time(),replacements=mapping))


if __name__=='__main__':main()
