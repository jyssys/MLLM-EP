# Online policy implementation

All policies use the same pretrained Base Ouro-1.4B weights and BF16 kernels.
Every real prompt and decode token executes Static R4.
No policy predicts a future token's completion time.
No policy changes model exit depth.
The policy ID `G1` means adaptive G-only. Its actual group size can be G=1,2,3,4.
This policy ID is distinct from the native Split path's fixed G=1 setting.

| Label | Action surface | Native execution path | Fairness |
| --- | --- | --- | --- |
| C1A | Adaptive C, G4 | Atomic | Completion-aware |
| C1S | Adaptive C, G4 | Staged | Completion-aware |
| G1 | Fixed model-tuned C, adaptive G | Staged | Completion-aware |
| J1 | Adaptive C and G | Staged | Completion-aware |
| M0 | Tuned fixed C | Native Mixed G1 | Stock |
| M1 | Same C as M0 | Native Mixed G1 | Completion-aware |
| F1A | Tuned fixed C, G4 | Atomic | Completion-aware |

C1A/C1S/G1/J1 share the frozen cost lookup and the fixed budget controller.
They share completion-aware fairness, selector infrastructure, and admission rules.
Each adaptive batch contains at most one prefill span.
The cost model estimates that span, not an unmodeled packed prompt batch.
Native fixed and Mixed controls retain native packing.
This difference is explicit and is not described as matched packing.
M0, M1, and F1A are unconstrained native controls, not budget-compliant adaptive policies.
Their B label identifies the paired comparison stratum only.
The primary matched-budget contrast is C1A/C1S/G1/J1.

C can change only when opening a fresh chunk.
A continuation retains its original actual token span.
G clips to the remaining depth and preserves native GPU-event dependencies.
The staged workspace reserves the maximum C2048 span for each adaptive request slot.
Native KV admission still checks capacity.
Preemption is disabled in this campaign.

The initial U2 utility divides eventual completed tokens by estimated remaining full-R4 stream service.
It pays prelude and every remaining recurrent group.
The first screen uses zero additional decode-yield penalty.
Thus U2 is a service-based heuristic, not a complete request-latency model.
CPU selection, metadata work, scheduling, and actual decode interruptions remain in measured E2E.
Their omission from the utility can reduce its effectiveness.

When no candidate fits, legal decode work has priority.
Idle progress and a 250-ms frontier-starvation escape can admit a marked overbudget action.
The fallback chooses the smallest predicted service action.
This can make a severely constrained C-only policy inefficient.
Large gains over such a fallback are not sufficient evidence of Joint's practical value.
Tuned G-only, unconstrained fixed controls, and native Mixed remain essential comparisons.

The TRAIN pilot covers all eight registered fixed budgets and both primary shapes.
It completed 32 jobs and 448 natural observer cohorts.
Its initial source versions include the documented planned-completion label correction.
The quiet confirmation uses only the corrected implementation.
It freezes B40/B120 and both shapes before reading FINAL_ONLINE outcomes.
It uses three independent model-process starts per shape/budget and two source cohorts per start.

The feedback controller exists as a tested definition.
It is not a measured online feedback result unless a separate executed plan says so.
No SLOWeave or P-PAS official implementation is claimed.
