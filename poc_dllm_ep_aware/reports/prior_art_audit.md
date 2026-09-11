# Prior-art audit

## Scope

The candidate claim under attack was narrow: use the *physical EP cost* of
optional dLLM inference actions when deciding whether to create those actions.
This is distinct from changing the base router, placing experts, or merely
executing already-required work faster.  The audit used papers and official
repositories available on 2026-09-12.

| Work | Work it changes | Distributed mechanism | Collision with this PoC | Consequence |
|---|---|---|---|---|
| TEAM | denoising calls and speculative branches | released code is single-GPU oriented | Candidate E is an EP-cost-aware tuning of TEAM speculation width | adjacent and potentially incremental |
| REFLEX | per-position expert budget | no verified public implementation found | Candidate B adds topology cost to its native marginal utility | plausible gap, but no faithful local quality result |
| DES | per-sequence expert coreset | memory/parallelism decoupling | Candidate D changes coreset selection using rank cost | very close to generic topology-aware coreset selection |
| Epoch | schedules compiled fresh/required dLLM work on EP | distributed execution of required work | does not decide which optional work is created | conceptual boundary remains, but faster communication shrinks our bound |
| TIDE | expert offload/placement for diffusion MoE | placement and transfer scheduling | not the same action space | rules out framing this as generic placement |
| DICE | staleness-aware overlap for diffusion MoE | communication/computation overlap | no independent >5% overlap window was measured here | generic overlap is not novel or supported |
| Generic EP routing/placement | maps routed work to devices | load, locality, and communication optimization | topology-aware action selection is adjacent | novelty requires a dLLM action/quality contract and material gain |

## Adversarial conclusions

### TEAM

The exact new observation is not that speculation has a cost.  It is that
TEAM's reduced NFE coexists with 32% more EP4 assignments/bytes and reverses
the winner on the measured reference EP substrate.  An EP-aware speculation
width is the most direct response, but width is already an obvious policy knob.
Without a quality-matched gain beyond static width tuning, this is
characterization rather than a successor method.

### REFLEX

REFLEX already treats expert work as optional and assigns variable k.  Adding
bytes or fanout to its score is a natural extension, not a new problem by
itself.  The only defensible gap would be a repeatable quality-equivalent case
where semantic marginal utility disagrees with calibrated distributed cost and
the joint policy recovers at least 8–12% request latency.  The local port could
not establish that quality contract and the generous oracle was only 3.83%.

### DES

DES already selects a smaller expert support.  Rank-aware coreset construction
is therefore close to a topology-aware variant of its own objective.  More
importantly, a 38/64 coreset still touched all four ranks at top-8, and even an
unrealistic one-rank-contact removal oracle was 3.90% E2E.

### Durability under better communication

The most implementable candidate, TEAM EP-aware width, falls from 4.01% to
2.01% if communication cost halves.  Thus an Epoch/DeepEP-class improvement
would make the already-small opportunity smaller.  This is a durability risk,
not a reason to claim orthogonality.

## Source status

- TEAM paper: <https://arxiv.org/abs/2602.08404>
- TEAM official repository: <https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM>
- REFLEX paper: <https://arxiv.org/abs/2608.01784>
- DES paper: <https://arxiv.org/abs/2602.00879>
- Epoch paper: <https://arxiv.org/abs/2609.09748>
- TIDE paper and repository: <https://arxiv.org/abs/2605.20179>, <https://github.com/ims-kdks/TIDE>
- DICE paper and repository: <https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html>, <https://github.com/Cobalt-27/DICE>

No public REFLEX, DES, or Epoch implementation matching the paper-faithful
execution contract was found during the bounded audit.  This is recorded as an
environment/evidence limitation, never as evidence that those methods fail.
