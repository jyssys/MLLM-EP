# Quality-safe EP-aware route-pruning summary

## Outcome

**NO-GO for practical P4 and FAIL for dLLM-specificity.** P4 was completed for
all 32 requests. Its 15.723% total simulated EP8
routed-MoE-stage reduction is dominated by a 14.377%
NFE reduction and comes with two additional wrong answers. The NFE-normalized
EP8 gain is 1.572%, only
0.578
points above P2, so the EP-specific 3-point gate is not met.

P2 1% is the only provisional gate-safe point (one additional wrong), but it
was not promoted to 128 because the matched P4 comparison failed. This is not
evidence for quality preservation at scale.

## Evidence boundary

- Quality and trajectories: actual GSM8K-32 model rollouts.
- EP4/EP8 times: EP2-calibrated simulation of routed-MoE stage only.
- Dense emulator wall time: deliberately not used as a latency result.
- HumanEval, temporal persistence, commit-aware policy, production runtime:
  not run by the pre-registered stop rule.

See `route_pruning_quality_gsm8k.md` and
`route_pruning_quality_ep_projection.md` for the complete tables.

QUALITY_SAFE_POINT_FOUND:
YES

BEST_SAFE_UTILITY_ONLY:
1% nominal / 0.998% actual, 30/32, EP4 16.496% total projected stage gain, EP8 16.078% total projected stage gain; provisional GSM8K-32 only

BEST_SAFE_EP_AWARE:
N/A; P4 1% failed with 29/32 and two additional wrong answers

EP8_EP_SPECIFIC_INCREMENT_AT_SAFE_POINT:
N/A

MAX_SAFE_REMOVED_ROUTER_MASS:
0.998% for provisional P2; 0% established for P4

AVG_K_AT_BEST_SAFE_POINT:
7.7529

EP8_MAX_MEAN_BEFORE_AFTER_SAFE_POINT:
1.2122 -> 1.2178 (P2 worsened)

EP8_WAIT_BEFORE_AFTER_SAFE_POINT:
17.017% -> 17.376% (P2 worsened)

HUMANEVAL_RUN:
NO

HUMANEVAL_QUALITY_DELTA:
N/A

TEMPORAL_LOW_MASS_PERSISTENCE:
N/A

TEMPORAL_HIGH_PRESSURE_PERSISTENCE:
N/A

TEMPORAL_JOINT_PERSISTENCE:
N/A

TEMPORAL_HEURISTIC_GAIN_OVER_CURRENT_P4:
N/A

COMMIT_AWARE_GAIN_OVER_CURRENT_P4:
N/A

DLLM_SPECIFIC_SIGNAL:
FAIL

PRACTICAL_METHOD_STATUS:
NO-GO

PRIMARY_BLOCKER:
P4 at 1% caused two additional GSM8K-32 errors and its NFE-normalized EP8 increment over P2 was only 0.578 percentage points.

NEXT_ACTION:
Retire the current no-renormalization P4 path; only revisit route pruning under a separately specified safety mechanism, otherwise prioritize exact balancing.

DO_NOT_IMPLEMENT_PRODUCTION_RUNTIME_AUTOMATICALLY:
true
