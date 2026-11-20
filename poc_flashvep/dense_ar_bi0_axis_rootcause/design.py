"""Freeze outcome-independent priorities and limited confirmation selection."""
import time
from .common import ROOT,write,initialize


def main():
    initialize()
    path=ROOT/'configs/frozen_research_design_v1.json'
    if path.exists():return
    write(path,dict(time=time.time(),base_commit='12e0f370a26ec544af81a90577874527ce8d4c5b',
        hypotheses=[
            dict(id='H_P',prediction='Token saves more source attention. Its advantage grows when source service is the bottleneck. Compare serial source costs and loaded queue decomposition.'),
            dict(id='H_D',prediction='Layer has a shorter destination suffix. Its advantage can grow when destination service plus decode residency is the bottleneck.'),
            dict(id='H_C',prediction='If source-window credit is causal, changing only its limit changes credit-full waiting and may alter the axis ordering.'),
            dict(id='H_G',prediction='Changing input bucket, chunk count or matched decode batch can change effective P/D service balance.')],
        primary=dict(dataset='LongBench-ShortAnswer200-v2',N=200,BI0=True,
            B=8192,S=1024,KV_GiB=42,KV_block=128,actual_batches=[1,1,1],source_credit=4,
            cuts=[2,4,8,12],QPS=[.4,.7,1.,1.2,1.65,2.,2.35],
            discovery_seed=2026112081,confirmation_seeds=[2026112071,2026112072,2026112073],
            numerical='Same-geometry state/greedy gates. Cross-geometry full200 frozen task-quality CI margin2points. No BI1 result is pooled.'),
        first_confirmation=dict(cut=12,QPS=2.35,seeds=[2026112071,2026112072,2026112073],
            reason='The user prioritized the previously unconfirmed discovery reversal.'),
        credit=dict(cut=8,QPS=2.35,values=[1,4,8,16,32],axes=['Native','Layer','Token'],
            actual_batches=[1,1,1],hidden_pool='Select a common fixed Layer capacity32,16,8 in that order using a worst-context memory/state preflight only. Capacity failures remain visible. A source credit above certified hidden capacity is not a valid Layer performance arm.',
            exclusion='A failed memory/lifetime arm is not eligible performance. Do not discard its raw logs.'),
        input_control=dict(targets=[8192,16384,32768],sampling='Use full original requests in deterministic predeclared length buckets. No new padding, truncation or performance-based resampling.',
            scope='Report actual min/mean/max and B8192 chunk counts. Nominal bucket centers are not exact generated input lengths.'),
        confirmation_selection=dict(maximum_extra_cells=3,
            rule='After the complete available discovery matrix, freeze the strongest Layer TTFT-ratio cell, the strongest Token TTFT-ratio cell, and the closest-to1 boundary cell among full200 quality-passing pairs. Do not select the same cell twice. Previous independent cut12 confirmation stays separate.',
            outcome_reporting='Repeat every selected pair on all held-out seeds regardless of sign. Failed or inconclusive confirmations stay in the main matrix.'),
        fixed_SLO=dict(TTFT_ms=5000,every_client_ITL_ms=34,request_TPOT_ms=33),
        decode_gate='Paired independent-start ITL p99 and TPOT p95 ratio CI upper bound <=1.05 vs the matching controlled Native.',
        budget_rule='No new large dispatch after start+9h. Stop expanding the grid if fewer than90minutes remain before that cutoff. Preserve at least1hour for closeout, figures, reports and push.',
        bridge='C3 Layer/Token must pass their own state and output tests. C0 evidence never qualifies a C3 arm. No active-decode insertion without a safe-window gate.',
        limitations='Finite cohorts are not sustainable capacity. Host service envelopes are not GPU-exclusive time. A single discovery start does not confirm a winner.'))


if __name__=='__main__':main()
