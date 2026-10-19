# Shared weights and selected computations: source audit

Revision 1. Decision: no changed assumption for C06 or C18.
This is a source audit, not a model experiment or a performance result.

## Identity and inspected scope

The primary record confirms Jiaju Wu, Yi Hu, and Muhan Zhang as the authors.
The latest observed version is arXiv:2609.39892v1, submitted at 2026-09-30 14:59:21 UTC.
The HTML body is available. The paper links its official repository.
No ResearchGate summary supports this audit. [Primary record](https://arxiv.org/abs/2609.39892)

The inspected paper sections are §§3–6 and Appendices A–H, with selected methods and limitations read directly.
This does not mean every result received an independent replication.

| Study | Model or intervention |
|---|---|
| Main graph experiments | Task-trained two-layer shared blocks. D8L6 uses width 256, four heads, and learned rank-48 maps. |
| Ouro letter-walk | Task-fine-tuned Ouro-2.6B, then a learned dense affine map. |
| Ouro cancellation | Two separately fine-tuned backbones, then learned rank-128 maps. |
| Additional Qwen evidence | Qwen3-8B's 36-layer stack, adapted for recurrence, then a learned affine map. |
| Synthetic knowledge graph | Task-trained two-layer block, width 256, eight heads, then a learned affine map. |

These are not demonstrations of unchanged Base checkpoints with a training-free exact executor.
Decoded answers and task accuracy do not certify identical internal transitions.
An eight-step request can exceed backbone training coverage but remain inside map training coverage.
The cancellation experiment disables adaptive halting. [Methods, Apps. C–H](https://arxiv.org/html/2609.39892v1)

## Official code: what actually executes

The inspected repository revision is `ec56df3fc548f3b0321475002ed302fc10169e35`.
The public archive contains source and saved results.
Its documentation excludes large original weights and some complete training infrastructure.
Saved-result reproduction is not a fresh model replay. [Reproduction limits](https://github.com/wjjpku/howloop/blob/ec56df3fc548f3b0321475002ed302fc10169e35/docs/REPRODUCIBILITY_LIMITS.md)

`train_full_reference.py:79–91` loads Ouro-2.6B with FP32 weights.
It enables backbone gradients, freezes `early_exit_gate`, and specifies four loops.
The loss uses BF16 autocast at lines 103–109.
This precision contract differs from the campaign's BF16 checkpoint execution.
Generation uses `exit_at_step=3` at line 130. [Backbone source](https://github.com/wjjpku/howloop/blob/ec56df3fc548f3b0321475002ed302fc10169e35/experiments/ouro_letter/train_full_reference.py#L79)

`train_affine_pair.py` verifies the task-trained backbone's update-200 checkpoint at lines 26–34.
Its `Affine.forward` computes `h + linear(h, delta, bias)` at lines 13–16.
The hook applies this learned map before loops after the first at lines 42–51.
The map adds a dense projection. It does not replace the backbone call.
The manifest explicitly records `gate_used=False` at line 67.
An identity-initialization check at lines 74–81 does not certify the map after training.
The code later updates its parameters. [Map source](https://github.com/wjjpku/howloop/blob/ec56df3fc548f3b0321475002ed302fc10169e35/vendor/remote/letter_walk_native_20260914/code/train_affine_pair.py#L13)

`parallel_semantics.py:41–69` first calls the original attention implementation.
It then records or replaces selected patterns, values, or outputs.
The pattern function explicitly computes scores and softmax at lines 47–53.
The replacement computes a pattern–value product at lines 65–69.
This is a causal intervention, not an attention-work avoidance certificate.
The observed source does not implement an exact sparse path for unchanged Ouro.
The word `parallel` in this filename does not establish distributed inference.
The source loads one CUDA model and executes paired conditions. [Patching source](https://github.com/wjjpku/howloop/blob/ec56df3fc548f3b0321475002ed302fc10169e35/experiments/ouro_256_mechanism/code/parallel_semantics.py#L24)

## Direct reference: fixed points and cyclic trajectories

The directly relevant reference is *A Mechanistic Analysis of Looped Reasoning Language Models*.
Its latest observed version is arXiv:2604.11791v1, submitted at 2026-04-13 17:55:36 UTC.
I inspected §4, §5.2, and Appendices C and D. [Primary record](https://arxiv.org/abs/2604.11791)

Proposition 4.1 assumes that the complete recurrent block reaches a fixed point.
It then derives fixed points for cyclic permutations of its layers.
It does not prove that a pretrained checkpoint reaches bitwise equality within four loops.
Proposition 4.2 bounds attention-pattern change by state change:

`||S(X_t) - S(X_(t-1))|| <= L_sm * 2B*kappa/sqrt(d) * ||X_t - X_(t-1)||`.

Small state change gives a bound on change, not an equality certificate.
The empirical reference point is the state after 128 recurrences.
The study includes pretrained Ouro-1.4B, Huginn-0125, and retrofitted Llama.
It explicitly reports that Ouro does not reach a strict fixed point in the inspected diagnostics.
Appendix D.3 also shows an Ouro trajectory that departs from apparent stability after recurrences 8–16.
These diagnostics cover selected inputs and metrics. They do not establish universal impossibility. [§4 and App. D](https://arxiv.org/html/2604.11791v1)

## Changed-assumption decision

The following conclusions are deductions for the campaign, not claims of either paper.

| Proposed transfer | Required changed assumption | Current decision |
|---|---|---|
| C06: cheap fresh residual | An exact, lower-cost representation for the current residual under the actual BF16 kernels. | A learned low-rank controller does not establish low-rank residuals for frozen projection weights. Keep `NO_NET_HEADROOM` for the audited dense formula. |
| C18: reuse a transition | Exact equality of its complete input and read state, with all required side effects preserved. | Equal decoded answers, similar attention, or stable PCA trajectories do not satisfy this condition. Keep `GENERIC_ONLY`. |
| Skip a fixed-point suffix | A certificate for the complete implemented transition, including depth-dependent readers and publications. | Neither inspected source supplies this certificate for campaign R4 or the original adaptive gate. |
| Use selected attention paths | An exact current-input mask or bound that authorizes skipped computation. | Patching patterns from another execution changes activations. It does not certify a native skip. |

An exact hidden-state match alone cannot authorize attention reuse.
The executor must also check every relevant KV prefix, mask, position, and runtime-dependent reader.
The unchanged adaptive gate still needs its original probability and CDF semantics.
Required terminal KV publication also remains part of the contract.
One cannot replace those checks with a task-level notion of the same graph node.

Finite precision does not remove the need for evidence.
BF16 rounding or underflow can produce exact coincidences that real-arithmetic similarity measures do not predict.
Such coincidences need conservative certificates for the actual implementation.
This audit neither rules them out nor observes new hits.

A successful certificate would still need comparison with generic memoization on the same tied unroll.
Hashing, comparison, storage, publication, and misses remain costs.
No source result supplies a net speedup for that candidate.

## Limits and time

This bounded audit follows one directly relevant reference. It adds no filler bibliography.
It makes zero search-engine queries. It uses direct primary retrievals and official repository metadata.
The source notes record failed retrievals, selected code ranges, and source versions.
No foreign code executes. No weights download. No GPU work or environment changes occur.
The audit creates no trained component and no new engine.

Actual segment start: 2026-10-19 12:45:16 UTC.
Validated evidence cutoff: 2026-10-19 12:51:33 UTC. Recorded interval: 377 seconds.
Only final metadata publication follows this cutoff.
This worker interval is not additional campaign elapsed time.
