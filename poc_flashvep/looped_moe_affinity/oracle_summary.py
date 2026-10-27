"""Report constrained trace accounting. Do not manufacture a wall-time oracle."""
from collections import defaultdict
import csv
import numpy as np
from .common import ROOT
from .aggregate_quality import csv_write

def read(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []

def main():
    placed=defaultdict(list)
    for row in read('placement_structural.csv'):
        placed[(row['model'],int(row['ep']),int(row['batch']),int(row['context']),row['method'])].append(row)
    generic={(r['model'],int(r['ep']),int(r['batch']),int(r['prefix_tokens'])):r for r in read('generic_control.csv')}
    baseline={(r['model'],int(r['ep']),int(r['batch']),int(r['prefix_tokens'])):r for r in read('baseline_e2e.csv')}
    strong={(r['model'],int(r['ep']),int(r['batch']),int(r['prefix_tokens'])):r for r in read('strongest_measured_generic.csv')}
    output=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [4,8]:
            for b in [1,2,4,8,16]:
                for length in [128,512,2048]:
                    key=(model,ep,b,length)
                    static=baseline.get(key,{});control=generic.get(key,{})
                    row=dict(model=model,ep=ep,batch=b,prefix_tokens=length,
                        static_e2e_ms=static.get('e2e_ms_mean','NOT_AVAILABLE'),
                        current_router_generic_e2e_ms=control.get('generic_e2e_ms','NOT_AVAILABLE'),
                        static_to_current_router_generic_speedup=control.get('static_to_generic_e2e_speedup','NOT_AVAILABLE'),
                        strong_production_generic_status='NOT_AVAILABLE',
                        loop_candidate_e2e_ms='NOT_IMPLEMENTED',generic_to_loop_e2e_speedup='NOT_AVAILABLE',
                        static_to_loop_e2e_speedup='NOT_AVAILABLE',
                        O0='NONIMPLEMENTABLE_TRAFFIC_LOWER_BOUND_ONLY',
                        O0_minimum_destination_ranks_per_token=1,
                        O1='FIXED_MEMORY_CALIBRATION_ONLY_HEURISTIC',
                        O2='FIXED_MEMORY_TRANSITION_AUGMENTED_HEURISTIC',
                        O3='UNATTAINABLE_PERFECT_ROUTING_NOT_A_LIVE_POLICY',
                        O4='BLOCKED_NO_VALIDATED_STRONG_GENERIC_WALL_TIME_MODEL',
                        prototype_gate='NOT_ESTABLISHED',evidence='HELDOUT_TRACE_ACCOUNTING_AND_SEPARATE_LIVE_CONTROL',
                        limitation='Destination fanout and load are not E2E costs. No ideal placement receives a fabricated latency.')
                    if key in strong:
                        actual=strong[key]
                        row.update(strongest_measured_generic_policy=actual['policy'],
                            strongest_measured_generic_e2e_ms=actual['e2e_ms_mean'],
                            matched_static_to_best_generic_speedup=actual['static_to_control_e2e_speedup'],
                            strong_bridge_generic_status='LIVE_MEASURED_EXACT_OUTPUT_ENUMERATED_CONTROLS',
                            O4='BLOCKED_NO_VALIDATED_LOOP_AWARE_PLACEMENT_WALL_TIME_MODEL')
                    for method in ['static','popularity_balanced','generic_coactivation','recurrent_transition_graph']:
                        values=placed.get((*key,method),[])
                        if not values:continue
                        n=sum(int(v['token_invocations']) for v in values)
                        for field in ['unique_destinations_mean','rank_load_cv','rank_load_max_mean']:
                            row[method+'_'+field]=float(np.average([float(v[field]) for v in values],
                                weights=[int(v['token_invocations']) for v in values]))
                        row[method+'_sampled_token_invocations']=n
                        row[method+'_original_remote_edge_fraction']=sum(int(v['original_remote_edges']) for v in values)/(2*n)
                        row[method+'_generic_source_remote_edge_fraction']=sum(int(v['generic_current_owner_remote_edges']) for v in values)/(2*n)
                    output.append(row)
    csv_write(ROOT/'placement_oracles.csv',output)

if __name__=='__main__':main()
