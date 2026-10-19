# Independent candidate search v4: certified value readiness

Worker: `/root/independent_axes`.
Segment began: 2026-10-19 12:32:46 UTC.
This block uses CPU algebra and primary-source inspection only.
It changes no frozen artifact or coordinator file.

## Outcome

This block explores one putative execution axis and adds no candidate card.
The hypothesis replaces operand-completion readiness with a certificate of the final BF16 output value.
Different unfinished sums can round to the same output.
Thus, exact output readiness can sometimes precede complete arithmetic.

The legal abstraction survives a constructed positive control.
Its proposed cross-normalization shortcut does not survive the dependency control.
Its real-arithmetic certificate also fails without a backend-specific floating-point enclosure.
No observed native-kernel tile hit rate or ownership advantage supports implementation.

The idea overlaps generic online arithmetic and C09's certificate principle.
Retargeting that principle to internal tiles is not enough to establish a new recurrence-specific axis.
The screened variant does not enter the shortlist.
This result does not prove that all exact value-dependent execution must fail.

The worker did not add a second hypothesis merely to increase the count.
The excluded families remain excluded.
This block does not tune TP, repeat failed hidden prediction, or reopen terminal KV aliasing.

## 1. Current evidence and stronger baseline

The worker read the frozen C01–C14 register and separate C15–C19 cards.
The worker also read the bounded actual-activation identity artifact.
That artifact reports no identical whole MLP rows in its selected cross-depth comparisons.
Its limited sampling does not estimate a population rate or test full attention-state equality.
The new hypothesis does not require such row equality.

The coordinator supplied these additional constraints:

- Native asynchronous scheduling is a stronger baseline than a host-synchronous implementation.
- P2 external graph events cover 97.8–99.0% of a local cached-decode window, not end-to-end execution.
- Tested fixed-R4 correction required four expensive rounds.
- Remote full-MLP execution lost to TP2/4 in the tested component conditions.
- HBM counters remain unavailable.

This block reruns none of those experiments.
It does not convert graph-event coverage into an end-to-end speedup bound.
It also does not infer traffic savings from encoded size, resident capacity, or operation counts.

## 2. Start from actual Ouro dependencies

The source uses D=2048, MLP width 5632, 16 MHA heads, and R=4.
The two Base sizes contain 24 and 48 physical layers.
The [semantics audit](ouro_semantics_audit.md) pins the official model sources.
This block also inspected the actual local CDB norm and block implementations.

For one block, the dependency sequence is:

`N1 -> Q/K/V -> RoPE/attention -> O -> N2 -> residual -> N3 -> gate/up -> SiLU/product -> down -> N4 -> residual`.

The end of each physical stack adds loop norm and the unchanged original adaptive gate.
All four block norms and the loop norm retain their actual predecessors.
The source contains no additional query/key norm.
Historical KV retains its request, depth, layer, position, and version.

Within an MLP, a final gate/up feature can feed SiLU and product independently of other features.
Ready product features can feed partial down-projection work.
Ordinary tile pipelines can already exploit completed features.
The proposed distinction is an earlier certificate of a completed value before all its dense terms execute.

## 3. Provisional hypothesis R4-A

For one output row, split the exact real dot product into prefix and unfinished tail.
Let p be a computed partial sum.
Let B bound the magnitude of every remaining exact product sum.
Let E enclose all differences between this computation and the actual target backend.

Then the target pre-cast value must lie in:

`I=[p-B-E, p+B+E]`.

Let Q denote the baseline's actual BF16 rounding operation at the selected tensor boundary.
If Q maps the entire interval I to one bit pattern, that output value is ready.
The worker can publish that value without guessing an unfinished contribution.
It can omit the remaining work only after the certificate completes.

This certificate preserves the baseline precision.
It does not replace BF16 with INT8, FP8, or fewer mantissa bits.
It also does not assume an earlier hidden vector remains valid.
The remaining products can be nonzero and can differ between valid completions.

If current x is already ready, a possible real tail bound is:

`B_m,k = ||W[m,k:D]||_2 * ||x[k:D]||_2`.

Weight suffix bounds can exist before the current token arrives.
Input suffix norms require the actual current x or another separately proved enclosure.
They cannot use a future recurrent hidden vector.
Every computed norm and stored bound needs conservative rounding.
Ordinary nearest-rounded metadata can underestimate a bound and invalidate the certificate.

The error enclosure E is not optional.
It must cover partial-sum arithmetic, the complete native reduction, and the target cast behavior.
NaN, infinity, subnormal handling, overflow, and signed zero need explicit rules or a fallback.
This screen does not supply that complete H100 proof.

## 4. What useful concurrency could exist

A bound worker can compute current-input suffix norms beside a projection worker's prefix contraction.
Both inputs are actual and ready.
After a valid certificate, an exact gate/up tile can feed the unchanged SiLU/product consumer.
A later down-projection can then process those certified product features.

This exposes a possible intra-pass value frontier.
It does not make all four recurrent passes ready simultaneously.
The next pass still needs its actual normalized predecessor and required original-gate routing.
The frontier is conditional on successful certificates, not on recurrence count alone.

A completed raw coordinate does not generally make its normalized coordinate ready.
For example, keep the first raw coordinate equal to 1.
Let an unfinished second coordinate be either 0 or 2.
The first normalized BF16 coordinate becomes 1.4140625 or 0.6328125 under the constructed source-order control.
Therefore, a local raw-tile certificate cannot bypass the global norm dependency.

An interval-aware norm might establish a stronger certificate in a special case.
That requires an enclosure for the remaining energy and every relevant cast.
The local dense certificate alone does not establish it.
No such complete norm or whole-pass certificate runs here.

## 5. Physical sharing and ownership

| Object | Readiness and owner | Required cost or restriction |
|---|---|---|
| Physical W tile | Model worker after load | Same immutable checkpoint revision |
| Weight suffix bound | Bound owner after sound preprocessing | Store and read conservative metadata |
| Current x suffix norm | Bound worker after actual x exists | Current-token work, not historical reuse |
| Prefix accumulator | Projection worker after executed terms | Preserve its numerical meaning and layout |
| Certificate | Worker after p, B, and E exist | Publish the exact target bit pattern or continue normally |
| Historical KV | Original request/depth/layer owner | No state merging or remote-history saving |
| Current KV | Original matching writer after all required Q/K/RoPE components exist | No provisional certificate may update committed KV |
| Gate policy | Original token owner after completed loop output | Same hazard, CDF, minimum depth, and maximum depth |

Shared physical weights permit one bound table to serve four passes.
They do not make the current input bounds equal across passes.
An alias-aware logical unrolling can retain the same single bound table.
An R1 executor can use the same value-dependent certificate.
No new ownership optimum follows merely from amortizing this metadata.

Keeping a bound service on another GPU adds input, certificate, and control messages.
Keeping it local competes for scalar units, registers, cache, and scheduling resources.
The native asynchronous baseline already removes avoidable host serialization.
A remote service must beat that baseline, not a CPU round trip introduced only for comparison.

Under feature TP, the final output depends on contributions from all relevant shards.
The certificate must combine their valid enclosures before publishing the final value.
A missing rank cannot count as a zero contribution.
Under output TP, each owner can certify its own output rows, but generic TP can use that same rule.
Neither layout establishes a recurrence-specific distributed advantage here.

## 6. Complete resource screen

For K chunks of 64 coordinates, one FP32 suffix bound per output row/chunk uses:

`metadata_bytes = output_rows * input_columns/64 * 4`.

Each of Ouro's three MLP projections gives 720,896 bytes under this illustrative layout.
That is 3.125% of its 23,068,672 BF16 weight bytes.
All three tables total 2,162,688 bytes per physical MLP.
These are structural capacity counts, not a measured implementation.
The count does not include error envelopes, indices, flags, or scratch.

Input suffix statistics add current-token reductions.
Certificate checks add work after selected contraction chunks.
Repeated p/B/E checks can interrupt the GEMM pipeline.
Publishing different rows at different times also creates masks and synchronization.
Misses pay these costs without omitting the original remaining contraction.

A scalar certificate does not automatically remove a whole Tensor Core operation.
The implementation must map successful certificates onto useful instruction or work-tile granularity.
If one required row remains uncertified, a whole-tile exit may remain unavailable.
Individual neuron counts cannot substitute for that mapping.

The method intends to omit arithmetic or advance exact consumer work.
It does not target dispatch overhead already covered by native asynchronous execution.
It does not claim HBM reduction without counters or a checked byte-trace model.
No latency estimate follows from these counts.

## 7. Primary nearby mechanisms

[DEX](https://arxiv.org/html/2610.11748), sections 2.1 and 6, describes online most-significant-digit-first arithmetic and output-dependent early termination.
Its exact policies concern ReLU-negative detection and sign-only head decisions.
Its calibrated pruning and low-digit skipping are approximate.
It uses quantized INT8 hardware, not unchanged BF16 Ouro on H100.

This is conceptual prior for readiness before complete arithmetic, not an equivalent BF16 implementation.
Ouro's SiLU also does not have ReLU's entire negative half-line of exact zeros.
The paper's exact ReLU shortcut cannot simply replace the actual MLP activation.
Its cycle and energy results are not evidence for this candidate.

[NVIDIA's PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
specifies at least single-precision accumulation for the relevant BF16 matrix instructions.
It leaves accumulation order, rounding, and subnormal handling unspecified in those instruction descriptions.
A generic sequential-FP32 proof therefore does not automatically qualify the native kernel.
This report does not identify or execute a candidate SASS path.

[Accurate Models of NVIDIA Tensor Cores](https://arxiv.org/html/2512.07004), section 4.1.6 and its numerical-model tables,
analyzes H100 numerical details and supplies software models.
The existence of those models reinforces the need for an explicit arithmetic contract.
This block does not install or run that software.
It does not copy its numerical parameters into an unvalidated native-operator certificate.

[Jia and Rinard](https://people.csail.mit.edu/rinard/paper/sas21.exploiting.pdf), sections 2 and 6,
show why real-arithmetic verification can fail under floating-point implementations.
Their recommendation is an implementation-aware numerical enclosure.
Their adversarial-verification setting is not this execution algorithm.

These sources constrain the novelty and correctness claims.
They do not establish a universal impossibility theorem for value-dependent execution.

## 8. Executable falsifier

Run from the campaign root:

`python scripts/value_readiness_falsifier_v1.py`

The [artifact](../artifacts/value_readiness_falsifier_v1.json) records five control groups and the metadata counts.

| Control | Constructed result | Scope |
|---|---|---|
| Nonzero unfinished tail | Prefix 1 with tail within ±1/1024 always rounds to BF16 1 | Positive real-to-BF16 certificate |
| Rounding midpoint | Tail within ±1/65536 can produce BF16 1 or 1.0078125 | Small tail alone is insufficient |
| FP32 accumulation | Products [2^24,1,-2^24] give real sum 1 and sequential-FP32 sum 0 | Real-sum certificate is not a backend certificate |
| Norm dependency | Same raw coordinate gives normalized 1.4140625 or 0.6328125 | Local certificate does not bypass norm readiness |
| Tile granularity | One certified row and one uncertified row do not certify the whole tile | No operation-count credit from a partial predicate |

The scalar accumulation example uses BF16-representable factors.
A standard sequential-FP32 gamma bound encloses its discrepancy.
That widened interval rejects the unsafe commit.
The scalar bound does not qualify H100 Tensor Core arithmetic.

The controls use exact fractions where possible and explicit software FP32 rounding where stated.
They contain no checkpoint weights, captured activation row, learned predictor, or GPU call.
They do not measure model quality, hit rate, gate agreement, or performance.
NaN, infinity, signed-zero, overflow, and FTZ cases remain outside this executable certificate.

The script SHA256 is `d01379a9a8078ab21e10faa83d7cd7638bacb3ee99cfd1a566b013146a3175ca`.
Actual execution ran from 12:36:56.340192 to 12:36:56.340991 UTC on 2026-10-19.
That tiny CPU runtime is not inference timing.

## 9. Both-size and both-mode screen

| Scope | Structural result | Empirical status |
|---|---|---|
| Base 1.4B fixed R4 | Same matrix/norm conditions, four input-dependent certificate opportunities | NOT_RUN |
| Base 1.4B original adaptive | Same conditions, original gate and KV routing remain mandatory | NOT_RUN |
| Base 2.6B fixed R4 | Same widths, twice the physical layers | NOT_RUN |
| Base 2.6B original adaptive | Same conditions, no new validation of Base adaptive behavior | NOT_RUN |
| System C=1 | Only conditional intra-pass feature work, no four-pass ready frontier | NOT_RUN |
| System C=2/4/8 | More independent rows, also available to matched generic batching | NOT_RUN |
| P2/P4 | Need actual shard-enclosure and native asynchronous controls | NOT_RUN |
| P8 | Same symbolic obligations, no eight-GPU performance claim | Projected only |

## 10. Decision and reopening condition

Do not promote R4-A to a twentieth card from this screen.
The simple cross-norm schedule is dependency-invalid.
The real-only exactness argument is numerically incomplete.
The valid local certificate remains a generic conditional-compute possibility with no demonstrated ownership advantage.

A reopening needs all of the following:

1. A sound enclosure for the pinned native BF16 numerical path, including its actual cast boundaries.
2. Actual whole-work-tile certificates early enough to omit useful work, not just individual scalar examples.
3. A legal consumer schedule that retains every norm, residual, attention, KV, and original-gate dependency.
4. A cost comparison against native asynchronous same-P TP with the same exactness contract.
5. A benefit that survives generic certificate and alias-aware-unroll controls.

Until those conditions hold, the next useful action is no engine build.
No numerical quality pilot or additional model rerun follows from this screen.
The report preserves earlier negative conclusions.

## 11. Search and time ledger

The [source observations](../artifacts/candidate_search_v4_source_observations.json) preserve the actual query strings and locators.
All searches use the mechanism rather than the Ouro model name.
The inspected sources are a bounded novelty check, not an exhaustive inventory.

The segment began at 12:32:46 UTC on 2026-10-19.
Source and axis derivation reached the 12:35:49 UTC checkpoint.
The executable controls reached the 12:36:56 UTC checkpoint.
Report synthesis and checks followed.
The block contains no TP tuning or new inference measurement.
The final evidence checkpoint was 12:40:31 UTC.
The active worker interval to that checkpoint was 465 seconds.
All substantive task work concerned the proposed value axis and its falsification, not TP tuning.
The report and source-observation prose fields passed the clarity checks.
Both JSON artifacts parsed, and the saved script identity matched.
Final publication followed that checkpoint.
Count the active worker interval once within concurrent campaign wall time.
Do not sum it with overlapping workers as extra elapsed time.
