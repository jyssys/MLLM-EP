# Split execution controls

Evidence type: source audit. No method latency result is reported here.

The recovered Atomic PD configuration requests `wavefront=false`.
The existing `CutConnector.create()` requires `wavefront_chunked_prefill=true` and scheduler mode `refill`.
`PartitionConnector` inherits that requirement for both Token and Loop Split.
The existing staged scheduler advances explicit token and recurrent-depth frontiers.
Changing this scheduler setting can change submission cost even at a full cut.

Therefore, a split comparison needs these controls:

1. Qualified native Atomic PD with its original scheduling setting.
2. Native full PD with the staged scheduler setting required by both connectors.
3. Full-cut snapshot PD with the same staged setting.
4. Partial Token and Loop Split with that same staged setting.

The first difference measures scheduling effects. The second difference measures transport effects.
Neither difference is a Loop-specific policy benefit.
All methods retain the same numerical fixes and prohibit Prefill/Decode recurrent-core co-batching.

## Token boundary restriction in the old policy

The old policy rounds its token prefix to `cfg['chunk']`.
With a full-prompt chunk, that calculation can turn a requested partial prefix into a full prefix.
The connector itself accepts an exact prefix position and limits the next scheduled task to that position.
The new policy adapter must therefore specify the prefix independently of chunk size.
A large chunk must not disable Token Split or silently change its requested ratio.

## Admission and state

The destination still needs full-request KV capacity. Partial work does not reduce the final Decode cache requirement.
Huginn Token Split also transfers the canonical suffix initialization state.
Loop Split transfers the whole prompt's intermediate hidden state and only the valid KV planes before its cut.
Source buffers remain owned until native transfer acknowledgment.

The prior 16-token D-local probe does not validate a whole-prompt recurrent continuation.
Large continuation graphs can require more workspace than the probe graph.
Any changed destination pool or graph geometry needs a matched full-PD control and renewed correctness checks.

## Entry points

- `adaptive_recurrence_cut/connector.py`: connector construction, valid-plane layout, and continuation import.
- `adaptive_recurrence_cut/staged.py`: explicit state rows and separate-phase scheduling.
- `adaptive_loop_token_pd_2gpu/connector.py`: exact prefix frontier and suffix import.
- `adaptive_loop_token_pd_2gpu/policy.py`: old prefix rounding and handoff recheck.
- `looped_pd_atomic_prefill/runtime.py`: actual scheduler, pool, and destination token budgets.

Method GPU tests remain gated by Huginn recovery and their own exact-state qualification.
