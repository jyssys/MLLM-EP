"""Freeze interpretation rules before the held-out crossover runs."""
from .common import *

def main():
    path=ROOT/'configs/bottleneck_and_slo_contract.json'
    if path.exists():return
    write(path,dict(time=time.time(),
        capacity_scope='Finite service cohorts with within-start repetitions. Sustainable capacity needs additional arrival/drain evidence.',
        rates=[.4,.6,.8,1.,1.2],screening_requests=24,selected_requests=64,selected_independent_starts=3,
        SLO_calibration=dict(source='Disjoint calibration prompts at low-rate native PD',
            TTFT_rule='2 times calibration request TTFT p95',
            TPOT_rule='2 times calibration request-average TPOT p95',
            ITL_rule='2 times calibration token-population ITL p99',
            scope='Frozen model-level thresholds before held-out runs. Primary TTFT plus request-average TPOT. ITL is secondary.'),
        P_compute_candidate=dict(offered_fraction_of_P_capacity_min=.85,
            D_fraction_of_max_legal_prepared_cache_capacity_max=.85,
            D_reservation_wait_share_max=.10,D_KV_blocked_observation_fraction_max=.10,
            decode_request_TPOT_p95_fraction_of_SLO_max=.90,
            P_compute_queue='Actual P receive to first GPU-work submission. Require growing backlog or increasing queue age.',
            qualification='Require kernel profile and service trace. NVML utilization alone is insufficient.'),
        KV_admission_limited='D credit refusal and significant waiting before reserve/P execution. Do not label this P compute queue.',
        stability='Finite 24/64 request cohorts demonstrate finite pressure and drain only. Do not claim unbounded sustainable capacity.',
        phase_C_gate='Verified P compute pressure plus credible D compute/KV/SLO slack. Otherwise GATE_NOT_MET.',
        source_cost_warning='Do not use queue-contaminated host TTFT as GPU service cost. Do not add overlapping durations.'))

if __name__=='__main__':main()
