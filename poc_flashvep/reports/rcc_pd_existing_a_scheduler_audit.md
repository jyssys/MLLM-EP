# RCC-PD: existing A and Native Mixed audit

## Scope and prerequisite

This audit uses the numerical release commit `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
The remote numerical branch contains this commit.
Both models passed the existing matched-runtime internal correctness gates.
The RCC mechanism still needs independent qualification.
Original-HF equivalence and benchmark answer accuracy remain separate, incomplete claims.

The RCC source spec exists at `rcc_pd_cross_depth_cobatching_poc_spec.md`.
Its SHA-256 is `2d316ac1db2d292446600095931b5517d2c512a7d807b38ecfd98ba59175d572`.
The source spec remains unchanged.
The current user instruction overrides its historical B status and its W2 output range.
W2 uses 128–256 output tokens if later stages pass.

## CPU source findings

Evidence class: `STRUCTURAL_BOUND` for source conclusions, not measured performance.
Paths below refer to the unchanged private RLT checkout unless a project prefix appears.

| Question | Finding | Source |
|---|---|---|
| How does A export state? | It transfers all prompt hidden rows and valid KV below the cut. It retains Huginn reinjection state. | `poc_flashvep/looped_pd_abc/split.py`, `prefill_submitted`, `_queue_snapshot` |
| What phase does D receive? | `PREFILLING_RECURRENT`, with a per-depth frontier initialized at the cut. | `split.py`, `accept_split` |
| Does default A fuse? | Default `refill` uses SplitPrefillPolicy. It selects continuation and decode phases separately. | `core/scheduling_policy.py` |
| Can existing A fuse? | Yes. Native `mixed` selects both recurrent phases in one scheduler output. | `MixedPolicy.schedule`, `Scheduler.take_phases` |
| Are depths required to match? | No. Rows must share a physical operation and execution-loop count. Logical depth stays in each span. | `worker/executor.py`, `build_execution_batch` |
| Are hidden rows combined? | Yes. Mixed gather packs continuation and decode hidden rows into one contiguous matrix. | `worker/model_runner.py`, `_mixed_hidden_sources`, `_mixed_hidden` |
| Are model calls combined? | ModelRuntime calls one `recurrent_prepared` operation per scheduled loop on that matrix. | `worker/executor.py`, `ModelRuntime.forward` |
| Are all GEMMs combined? | Not necessarily. Qualified Down and Huginn adapter projections use 256-row tiles internally. | Qualified `patches.py` |
| What happens to attention? | Packed metadata separates request/depth sequences. FA4 supports causal variable-length queries over paged KV. | `attention/metadata.py`, `kv_cache.py`, `kernels/flash_attention.py` |
| Does this prove exact output? | No. Packed attention changes batch geometry. New token and state tests remain mandatory. | Numerical release limitations |
| Does Huginn fit? | Its state carries state plus injection. Its synchronous runner supports staged prefill under an explicit guard. | `models/huginn.py`, `engine/stage_engine.py` |
| Can A migrate long multi-chunk prompts now? | No. `reserve` rejects prompts longer than the configured chunk. Removing the guard alone is unsafe. | `split.py`, `reserve` |

## Implementation distinction and novelty

The proposed fixed-cut fused arm already exists as **A plus Native Mixed** at source level.
This PoC names that existing arm `A_native_mixed`.
It does not create a second name and claim an independent implementation.
The `A_sep` arm uses the same split connector and cut with the default separate phase selection.
`PD_G1` retains full P-side prefill and staged G1 execution.

A GPU probe must test whether eligible rows actually reach the combined path.
It must record row offsets, request IDs, logical depths, physical operand shapes, and actual matrix operations.
A scheduler label alone does not prove fusion.

Conditional admission could add a control mechanism.
However, existing load-aware prefill deflection already considers spare D capacity.
An admission controller therefore needs measured incremental benefit and a precise distinction from this prior work.
No controller is justified merely because A can call the existing mixed runner.

## Model compatibility matrix

| Property | Ouro | Huginn |
|---|---:|---:|
| Fixed recurrent depth | 4 | 32 |
| Initial cut | 2 | 16 |
| Physical recurrent layers | 24 | 4 |
| Prelude / coda layers | 0 / 0 | 2 / 2 |
| Core width | 2048 | 5280 |
| Continuation carrier width | 2048 | 10560 |
| Attention heads / KV heads | 16 / 16 | 55 / 55 |
| Head dimension | 128 | 96 |
| Maximum model context | 65536 | 4096 |
| Qualified projection repair | Down + small-row RMSNorm | Down + adapter + small-row RMSNorm |
| Async scheduling | Retained | Unsupported by this runner |
| Whole-prompt split transfer | Existing | Existing guarded adapter |
| Long multi-chunk continuation | Not qualified | Not qualified |

Huginn cannot accept a 4096-token prompt plus 512 new tokens under its pinned context contract.
With 512 new tokens, its prompt limit is 3584 tokens.
Later workloads must record this limit rather than extend RoPE or silently truncate prompts.

## Dependency and state lifetime

The source records an event after the prompt-state copy.
Transfer submission waits for that event.
The receiver accepts only after every committed transfer sequence completes.
The imported state is a view of the registered receive slab.
ModelRunner copies that state into its own prompt buffer before use.
It synchronizes the admission copy before deleting the borrowed view.
The receiver retains its transfer entry until the request leaves the scheduler.
The source retains its slot until remote acknowledgement and local transport completion.
These source dependencies are consistent with safe ownership.
They are not substitutes for a byte-identity GPU test.

## Fairness hazards and controls

1. The same numerical repair must apply to every arm.
2. Down and adapter padding can create apparent fusion savings.
3. Actual 256-row matrix counts must accompany logical row counts.
4. Decode graphs cover eight rows in the initial matched configuration.
5. A mixed prompt-plus-decode batch can exceed that graph limit and run eager.
6. Common-eager diagnostics must separate graph availability from grouping effects.
7. A native graph comparison remains necessary for any serving claim.
8. Heavy operand tracing cannot provide clean serving latency.
9. A full D admission queue can prevent fusion even when the source supports it.
10. A real mixed execution must contain both continuation and decode spans.

The first probe uses 128-token prompt prefixes, then 256 tokens if correctness permits.
These are mechanism tests, not the proposed representative W1 workload.
They use frozen ShareGPT token prefixes from the previous study.
Prefix extraction can remove prompt suffixes and is explicitly diagnostic.
Natural EOS remains enabled unless a configuration explicitly requests controlled generation.

## Stage decision before GPU execution

`ALREADY_PRESENT` applies to the fixed-cut co-batching code path.
Physical fusion and performance remain `NOT_MEASURED` at this audit checkpoint.
No new RCC kernel or distributed engine is required for the first test.
The next step is a small two-GPU operator and output probe for each model.
No admission controller, long-context migration, or capacity campaign starts before that gate.

## Prior-art boundary

DistServe measures goodput under joint TTFT and TPOT constraints.
It establishes P/D disaggregation as prior work.
Source: [DistServe, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).

SARATHI combines chunked prefill and decode work to improve utilization.
Prefill/decode co-batching itself is therefore not new.
Source: [SARATHI, Microsoft Research](https://www.microsoft.com/en-us/research/publication/sarathi-efficient-llm-inference-by-piggybacking-decodes-with-chunked-prefills/).

DuetServe chooses aggregation or separation within a GPU under latency constraints.
Its source does not establish cross-GPU recurrent continuation as a new contribution here.
Source: [DuetServe, ICML 2026](https://proceedings.mlr.press/v306/gao26m.html).

Load-aware prefill deflection sends prefill work to D capacity.
Any RCC admission claim must distinguish its state-transfer and partial-recurrence mechanism.
Source: [Load-aware Prefill Deflection](https://arxiv.org/abs/2607.02043).

These comparisons identify overlap. They do not establish novelty or measured speedup.
