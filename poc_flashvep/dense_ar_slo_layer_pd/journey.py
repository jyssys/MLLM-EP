"""Keep tested hypotheses separate from method performance claims."""
from .common import *


def main():
    entries=[
        dict(Hypothesis='Long-output migration0 is caused only by the zero admission-wait limit.',
            Evidence='Eighteen frozen runs replay exactly. Raising only that limit through1000ms still selects zero migrations.',
            Code_Change='Candidate-level overlapping gate replay and one-variable counterfactuals.',
            Cost='CPU trace analysis only.',Result='The wait limit is not the sole blocker. Removing both wait and known-ingress predicates exposes608 Output64 and268 Output200 request candidates.',
            Decision='Replace the heuristic only after explicit affected-work deadline checks. Counterfactual candidates are not physical or online capacity.',
            evidence='TRACE_ANALYSIS_AND_OFFLINE_ORACLE'),
        dict(Hypothesis='Full-prompt text priming blocks the same worker that delivers Decode.',
            Evidence='All60 greater-than30ms worker intervals in the selected diagnostic overlap ADD_REQUEST. ADD wall costs are about9–12ms.',
            Code_Change='Verified complete-ASCII ByteLevel suffix priming with an exact fallback.',
            Cost='CPU equivalence tests and N32 full-model state qualification.',
            Result='The CPU text and GPU full-state gates pass.',Decision='Apply this generic improvement to every arm. Do not count it as a Layer-specific gain.',
            evidence='MEASURED_HOST_BOUNDARIES_AND_CORRECTNESS_ONLY'),
        dict(Hypothesis='Phase-correct short Prefill normalization pays avoidable padding and copy costs.',
            Evidence='The independent GPU test compares210 BF16 cases against the original padded operation.',
            Code_Change='Reuse the scalar fused normalization body with the original256-thread Prefill launch.',
            Cost='One small normalization extension. No GEMM or Attention changes.',
            Result='Every tested output and residual is byte equal. Full-model qualification also passes.',
            Decision='Apply the same exact correction to all arms. Micro event enclosures are not Serving speedups.',evidence='CORRECTNESS_ONLY_AND_ISOLATED_GPU_EVENTS'),
        dict(Hypothesis='Sorting KV descriptor indices alone removes fragmented transaction overhead.',
            Evidence='The sorted path still reports about18000 native descriptors for a full8k prompt.',
            Code_Change='Compare index sorting with actual physical byte-range coalescing.',
            Cost='Keep original registration, notifications, ACK and memory ownership.',
            Result='Physical coalescing reduces some18000-descriptor transfers to36. All cuts pass exact state tests.',
            Decision='Evaluate transport separately with A/B/A. Do not infer an end-to-end gain from fewer descriptors.',evidence='CORRECTNESS_ONLY_AND_MEASURED_HOST_BOUNDARIES'),
        dict(Hypothesis='Continuous cached-query destination admission is safe after coalescing.',
            Evidence='The N64 Output16 diagnostic has ITL p99 about68.37ms despite exact outputs.',
            Code_Change='Optional native_continuous_D_ingress diagnostic only.',Cost='One isolated diagnostic run per tested output cap.',
            Result='Decode-relative safety is not established.',Decision='Keep the whole-cohort guard for primary Layer execution. Exact causal shares remain NOT_IDENTIFIED.',evidence='MEASURED'),
        dict(Hypothesis='The installed standard async scheduler can strengthen the Colocated frontier.',
            Evidence='N32 all-layer KV and final Hidden qualification passes. Output16 S1 discovery ITL p99 is about24.42ms.',
            Code_Change='Enable the existing async_scheduling option. Keep original BF16 and exact phase correction.',
            Cost='No custom colocated scheduler.',Result='The initial async discovery points improve several previous points.',
            Decision='Retune Colocated independently for each SLO objective. Replace weaker queued long tests before they start.',evidence='CORRECTNESS_ONLY_AND_MEASURED'),
        dict(Hypothesis='A source macro can feed multiple smaller destination cohorts without a new ownership manager.',
            Evidence='The old all-pending predicate becomes false after its first destination cohort finishes.',
            Code_Change='Accept monotonically advanced decode/finished states at the already-complete source frontier.',
            Cost='Charge every intercohort Decode residence in the completion model.',
            Result='CPU regression tests and N32 actual four-by8 destination-cohort full-state tests pass.',
            Decision='Test the same source and destination geometry in Controller Zero, Static and Dynamic.',evidence='CORRECTNESS_ONLY'),
        dict(Hypothesis='A finite-cohort drain rate is sufficient to establish sustainable capacity.',
            Evidence='Initial240-second pilots fail the queue-stability screen even when latency improves.',
            Code_Change='Separate warm-up,600-second measurement and follow-through. Keep every offered arrival.',
            Cost='Long isolated runs and independent-start confirmation.',
            Result='Native Output16 at nominal2.8 QPS passes the first ten-minute Balanced and Decode-priority screen.',
            Decision='Treat this as one qualified tested point, not a maximum or a held-out result.',evidence='MEASURED'),
    ]
    write(ROOT/'analysis/optimization_journey.json',dict(entries=entries,
        scope='This journal records hypotheses and stopping decisions. Per-run immutable configurations and raw receipts remain authoritative. No unmeasured GPU idle or method superiority is claimed.'))


if __name__=='__main__':main()
