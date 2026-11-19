"""Compare standard and coalesced Native transport on identical natural traces."""
import json
import time
from .common import ROOT,write
from .campaign import serving


def main():
    declaration=ROOT/'configs/NIXL_latency_AB_design_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),ranked_hypotheses=[
            'H1: Many small descriptors block the D host in transfer setup. Exact byte-range coalescing reduces host setup and ITL spikes.',
            'H2: Transfer or stream interference dominates while host setup is small. Host timestamps alone do not identify pure DMA cost.',
            'H3: Runtime JIT dominates spikes. Only one D JIT event was observed in the initial N32 control, so this does not explain most long steps.'],
            workload='capacity_pilot40.json',rates=[.5,2.],arrival_seed=2026112012,
            sole_execution_change='Exact physical KV byte-range coalescing. Keep sampling, model math, B/S, request order, arrival trace, output cap, native completion and ACK.',
            scope='Discovery A/B, one independent start per cell. Do not claim a significant speedup or long-run capacity.'))
    gate=ROOT/'analysis/coalesced_Native_N8_full_state_gate.json'
    while not gate.exists():time.sleep(2)
    assert json.loads(gate.read_text())['passed']
    values=[]
    for rate in (.5,2.):
        baseline=ROOT/'analysis'/f'pilot_Native_LongBench_N40_q{rate:g}_BI1_v1_metrics.json'
        while not baseline.exists():time.sleep(2)
        name=f'pilot_Native_COALESCED_LongBench_N40_q{rate:g}_BI1_v1'
        actual=serving(name,'Native','capacity_pilot40.json',40,rate,2026112012,
            phase='NATIVE_TRANSPORT_AB',nixl_manual_coalesce=True)
        original=json.loads(baseline.read_text())
        values.append(dict(name=name,baseline=original['name'],offered_QPS=rate,
            baseline_metrics={k:v for k,v in original.items() if k not in ('per_request','by_task')},
            candidate_metrics={k:v for k,v in actual.items() if k not in ('per_request','by_task')},
            evidence='MEASURED_DISCOVERY_AB'))
        write(ROOT/'analysis/Native_transport_AB_v1.json',dict(rows=values,
            scope='One start per arm. Exact output and normal credit retirement required. Source and per-request logs preserve all failed attempts.'))


if __name__=='__main__':main()
