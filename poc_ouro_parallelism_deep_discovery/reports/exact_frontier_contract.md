# Exact operator frontier and ownership contract

## Scope and result

This contract describes Base Ouro-1.4B and Base Ouro-2.6B.
Their physical layer counts are 24 and 48. Both use four recurrent passes.
The adaptive reference uses the original gate, q=0.5, minimum R2, maximum R4, and `last_exited`.
Prefill remains full R4.

The unchanged operator graph contains useful local forks and independent request frontiers.
It contains no edge that supplies the next loop's complete hidden state before the current loop's boundary norm.
This is a statement about the inspected graph, not every possible exact or quality-controlled inference algorithm.

A specific exact overlap exists before minimum depth.
R2 is mandatory regardless of the R1 gate under minimum R2.
R2 can therefore execute while the runtime transfers or accumulates the R1 hazard.
At depths where exit is legal, next-pass execution requires a provisional-state contract until the gate resolves.

No GPU execution occurs in this analysis.
The CPU graph checks establish only the stated symbolic dependencies.
They establish no numerical parity, physical overlap, latency, or quality result.

## Node identities and readiness

Each operation has identity `(request, generation, token, recurrent_pass, physical_layer, operator, version)`.
Each operation also has an owner, tensor layout, weight identity, and explicit read/write addresses.
Readiness requires every predecessor's output and every required state version.
Readiness alone does not provide a free compute, memory, or communication resource.

Let C denote completed nodes. Let A denote allocated weights and state.
Let R denote the available resource leases.
The legal frontier contains a node v only when these conditions hold:

```text
predecessors(v) are contained in C
all required weight/state versions are resident or their transfers completed
all reads match the request, token, depth, layer, and branch version
every write has an exclusive lease
the required compute/copy/collective resource belongs to R
the launch cannot publish an unresolved branch as authoritative state
```

Two ready nodes can overlap only if they satisfy both data and resource conditions.
The same device can still serialize mathematically independent nodes.
Two devices can still contend for the same link or collective group.

For a declared node cost c(v), a feasible schedule must satisfy these constraints:

```text
finish(v) = start(v) + c(v)
start(v) >= max(finish(p) for p in predecessors(v))
start(v) >= each required weight/state readiness time
overlapping resource leases cannot exceed that resource's capacity
```

The longest weighted dependency path supplies a conditional latency lower bound.
Resource contention can increase latency beyond that path.
No measured node costs or physical latency prediction enter this contract.

## The block frontier

For one request and token, the semantic block graph is:

```text
x -> N1 -> {Q, K, V}
Q -> RoPE_Q
K -> RoPE_K
{RoPE_K, V} -> current_KV_write
{RoPE_Q, current_KV, required_historical_KV} -> attention
attention -> output_projection -> required_TP_reduction -> N2
{x, N2} -> attention_residual -> N3
N3 -> {MLP_gate_projection, MLP_up_projection}
MLP_gate_projection -> SiLU
{SiLU, MLP_up_projection} -> product -> down_projection
down_projection -> required_TP_reduction -> N4
{attention_residual, N4} -> block_output
```

N1, N2, N3, and N4 are four separate RMSNorm modules.
The two reductions appear when the corresponding row-parallel projection uses TP greater than one.
They are identity operations at TP1 in this symbolic representation.

The output projection must include every required attention contribution before N2 consumes its complete result.
The down projection must include every required MLP contribution before N4 consumes its complete result.
The nonlinear norms and SiLU prevent arbitrary reuse of unreduced partial values.

The native vLLM implementation stores part of each residual separately.
Its next norm can combine that residual with the current hidden tensor.
The semantic block output above includes both parts.
A runtime must not treat the stored hidden tensor alone as that complete output.

Source locations are official `modeling_ouro.py:401–423` and native vLLM `ouro.py:260–281`.
Native vLLM applies its boundary norm to hidden plus residual at `ouro.py:367–374`.

## Recurrent boundary and original gate

Let z_r denote the last physical block's complete output at pass r.
Let h_r denote its normalized loop output.

```text
z_r -> loop_RMSNorm -> h_r
h_r -> original_gate_logit -> sigmoid_hazard -> cumulative_exit_state
h_r -> next_pass_N1, subject to the execution guard
h_r -> LM_head, subject to the selected-output guard
```

The original gate consumes h_r, not z_r or a predicted replacement.
The next pass also consumes h_r.
Thus the gate and next-pass input norm have a common available data input.
Their control guards determine whether both belong to the committed execution.

For fixed R4, passes R1 through R4 are mandatory.
The output uses h_R4.
Unused exit heads need not determine this output.
The inspected native vLLM forward never invokes its declared gate.

For adaptive execution, the policy accumulates each hazard in order.
The R1 hazard contributes even though minimum R2 prevents an R1 exit.
R2 can execute before host consumption of the R1 hazard.
The R2 exit decision must still use the actual R1 and R2 hazards.

At eligible depths, a nonprovisional next-pass launch waits for the current continuation decision.
At R4, the maximum-depth rule forces exit.
The final gate is unnecessary for this routing decision, although a runtime may still compute it.

Source locations are CDB `exit_policy.py:141–169` and `continuous_api.py:1656–1682`.
The selected synchronous wrapper remains unchanged in this analysis.

## Exact provisional next-pass work

An exact runtime may execute next-pass work after h_r exists but before the current gate resolves.
This is control-flow speculation with a known input, not stale-state prediction.
The following requirements preserve the selected same-step policy:

1. Retain immutable h_r for the possible exit and its LM head.
2. Read only committed historical KV and the current token's declared provisional rows.
3. Give the next pass a separate hidden version and exclusive append-row leases.
4. Do not publish the next pass's rows or output as the request's selected result.
5. If the gate continues, accept the provisional work under the continuation decision.
6. If the gate exits, discard that work and select h_r.
7. Publish terminal-depth KV from r into deeper logical slots before a successor reads them.

The scratch requirement concerns new current-token rows. Immutable history need not be cloned in full.
An ordinary reserved deeper row can serve as provisional storage only under an exclusive unpublished lease.
A losing provisional row cannot replace the selected terminal row in a later token's history.
Cancellation, extra compute, extra hidden storage, state restoration, and synchronization all count as added cost.

The current CDB hidden bank overwrites the active slot after each recurrent launch.
See `model_runner.py:689–693`.
Exact provisional execution therefore needs a retained terminal-hidden version or equivalent double buffering.

The existing delayed-consumption mode does not implement this contract for the selected same-step policy.
It consumes a previous gate after the current pass and routes the current pass to coda.
See `continuous_api.py:1116–1122,1232–1246`.
That policy can change executed depth and selected output.
Enabling it is not an exact same-policy overlap comparison.

## Token and KV publication frontier

The next input token requires the preceding selected logits and sampling decision.
It does not require a host readback when the next consumer can use device-resident token IDs.
It never becomes available merely because a later recurrent pass is ready.

The embedding lookup does not read KV.
After the token decision, embedding can potentially overlap pending historical-state publication.
Its first attention consumer must wait for every historical KV version it reads.
This distinction avoids an unnecessary global state barrier before a pure lookup.

For fixed R4, each historical `(token, depth, layer)` state remains distinct.
For adaptive `last_exited`, computed shallow states remain distinct.
Only deeper uncomputed slots receive the terminal-depth state.
Terminal copies or equivalent aliases must precede the relevant later read, not unrelated requests' work.

The initial contract allows one recurrent token per request in flight.
It permits any number of independent requests subject to capacity and resource leases.
Multiple in-flight tokens of one request require an additional versioned causal protocol.
LoopSpec or another speculative method with a verification contract can supply different work.
Its guarantees and costs remain separate.

## Legal overlaps and nearest standard mechanism

| Ready work | Exact condition | Main added cost or restriction | Classification |
|---|---|---|---|
| Q, K, and V projections | N1 output exists | Splitting a fused QKV GEMM can reduce efficiency | Generic operator/TP parallelism |
| Q and K rotary transforms | Their respective projections exist | Extra launches and data layout | Generic operator overlap |
| MLP gate and up projections | N3 output exists | Native path already fuses their GEMM | Generic operator fusion/TP |
| SiLU and remaining up projection work | The relevant gate tile exists | Product waits for both operands | Generic tile pipeline |
| Attention heads or sequence partitions | Queries and exact required KV exist | Output merge, softmax merge, communication | Generic TP/CP |
| Historical KV transfer and earlier independent compute | The history already passed commit | Capacity, traffic, and contention | Generic prefetch/disaggregation |
| R1 gate transport and mandatory R2 | Minimum depth is R2 | Preserve R1 hazard for the R2 decision | Exact adaptive prefix scheduling |
| Eligible gate and provisional next pass | h_r exists and each state has a version | Wasted work, retention, discard/commit | Generic guarded execution, recurrence-specific placement possible |
| Gate and provisional LM head | h_r exists | Extra head work if that depth does not exit | Generic conditional-work speculation |
| Different requests' ready operations | No cross-request state dependency | Shared worker and communicator contention | DP, CDB, or operator pooling |
| Different causal prefill chunks | Each operator's earlier-token KV is ready | Fill/drain, messages, chunk shapes | Generic causal PP unless another property changes the optimum |
| Sampled token lookup and pending unrelated copies | The selected token exists | Attention still waits for its own required history | Generic device-side re-entry |

The table describes legal transformations, not completed overlap implementations.
The runtime can transfer or prefetch committed historical KV before a current query exists.
Query-dependent attention scores still require the current query.
The current fused FA3 append/read path controls its own write/read order.
A write-behind replacement must make current K/V available directly and preserve later commit dependencies.

## B1 and independent-request cases

At B1 decode, the committed graph has a recurrent chain across the physical stack.
Its local Q/K/V, MLP, head, and transport forks remain available.
Replicating weights does not remove the pass-boundary dependency.
Provisional continuation can overlap gate latency but can add unused work.
No single-request speedup follows from the number of idle GPUs.

With independent requests, one request's pass r can be ready while another request executes a different pass.
Their common physical weights can permit grouped execution.
Their KV addresses must still retain each row's own depth and request.
This frontier can exist under fixed R4 through staggered arrivals or different request progress.
It does not require early-exit heterogeneity.

A synchronized fixed-R4 cohort can remain phase-aligned.
It need not provide mixed-depth ready work.
Creating artificial delay to produce a wider frontier can remove the expected benefit.
A scheduler must count that delay and compare against a competent same-resource baseline.

Independent DP replicas need no shared decode-step barrier.
The CPU graph examples therefore contain no cross-request data edges.
Requests on one compute owner still compete for that owner's execution capacity.

## Actual CDB stream boundaries

CDB allocates one compute stream and one D2H stream on CUDA.
See `continuous_batching/input_outputs.py:518–524`.
The recurrent stage executes its model operations on the compute stream.
Its graph capture and replay use that same stream at `model_runner.py:908–938`.
This source does not implement simultaneous physical layers or recurrent passes on separate compute streams.

The selected same-step path synchronizes the compute stream before host gate routing.
See `model_runner.py:695–699`.
The synchronous-signal predicate does not exempt the mandatory minimum-depth prefix.
See `exit_policy.py:141–148`.
Thus the source contains a removable host boundary before mandatory R2, subject to an exact hazard-retention implementation.

The optional gate readback records a compute event and makes the D2H stream wait for it.
It then records a host-copy completion event.
See `model_runner.py:796–821`.
The coda path uses the same compute-then-D2H event pattern at lines 1007–1034.
Its source explicitly states that GPU model work uses one lane.
Asynchronous host handling is not evidence of simultaneous recurrent GPU compute.

Terminal copies execute on the compute stream before later reads.
See `continuous_api.py:1210–1230`.
Their source order is stronger than a universal cross-request requirement.
A different schedule must preserve the actual per-request read dependencies and copy-resource constraints.

## Native vLLM collective boundaries

The inspected native model uses column-parallel QKV and gate/up projections.
It uses row-parallel attention-output and MLP-down projections.
`RowParallelLinear.forward:1688–1697` computes a local result and then invokes TP all-reduce when enabled.
The complete result feeds the following RMSNorm.

For fixed R4, this gives 192 row-projection reductions per token at L24 and 384 at L48.
These are structural counts for TP greater than one.
They exclude embedding reduction, logits gathering, and any optional gate operation.
They are not measured latency or an achievable savings budget.

`communication_op.py:12–14` routes the reduction to the current TP group.
`CudaCommunicator.all_reduce:273–338` chooses among supported collective implementations.
The actual path depends on runtime settings, tensor shape, and supported hardware.
This source audit does not claim which path a new run would select.

The PyNccl path submits NCCL on the current CUDA stream at `pynccl.py:186–196`.
That submission does not imply a host-wide barrier.
The dependent norm still requires the reduction result on device.
The graph-capture context waits for initialization and captures on its selected stream at `parallel_state.py:632–639`.

One TP group's ranks must issue compatible collectives in the same operation order.
Interleaving request tasks differently on each rank can violate that contract.
Separate request readiness does not authorize inconsistent collective ordering.
Concurrent communicators also need explicit resource and synchronization accounting.

The native logits processor gathers the projected vocabulary shards at `logits_processor.py:138–153`.
The next token decision therefore requires the declared sampling interface, not one arbitrary vocabulary shard.
The inspected Ouro forward supports fixed depth, not online original-gate adaptive execution.

## Ownership and lifetime

| Resource/state | Owner | Readiness and lifetime rule |
|---|---|---|
| Physical weight shard | Model or operator worker | Resident before use, immutable across requests and passes |
| Hidden/residual pair | Current compute owner and version | Publish the complete semantic value before its consumer |
| Historical KV | Declared request/depth/layer owner | Immutable after commit, retained until all readers finish |
| Current append row | Exclusive request/token/version lease | No duplicate writes or uncommitted foreign reads |
| Gate CDF and survival | Request/token policy owner | Consume actual hazards in recurrent order |
| Provisional hidden/KV | Branch version owner | Never publish a losing continuation |
| Sampling state | Request output owner | Preserve greedy selection or the declared RNG-counter semantics |
| Compute lane | Device execution owner | Readiness does not establish concurrent capacity |
| Copy path | Source/destination copy owner | Retain buffers until the completion event |
| TP collective slot | Shared TP group order | Match rank participation, shape, dtype, and operation ID |

For sampling rather than greedy decoding, request-local RNG ownership also matters.
Changing request order must not silently change the claimed sampling contract.
This analysis performs no distribution-preservation test.

## Beyond the unchanged operator DAG

An algebraic reorganization can change the graph without using stale inputs.
For RMSNorm, write `N(x)=gamma*x/rho(x)` in real arithmetic.
A following linear map can form `W(gamma*x)` while the runtime computes rho(x).
It then scales that numerator by `1/rho(x)`.

This permits a numerator/statistics fork outside the unchanged operator ordering.
Adjacent loop-boundary and next-input RMSNorms also admit a combined real-arithmetic expression.
Neither fact removes the required denominator or residual dependence.
The official BF16 intermediate casts prevent an automatic bitwise-equivalence claim.
Such a transformation belongs to the numerically reordered track and needs matched checks.

For completeness, let a and b denote the loop and next-input norm weights.
Let e0 and e1 denote their respective epsilon values.

```text
m0 = mean(x*x)
m1 = mean((a*x)*(a*x))
N_input(N_loop(x)) = (a*b*x) / sqrt(m1 + e1*(m0+e0))
```

The expression assumes real arithmetic and positive norm denominators.
It does not reproduce the original sequence of BF16 rounding operations.

This example limits the scope of a dependency rejection.
An invalid schedule for the current DAG does not prove every algebraically reorganized method impossible.
It also does not exclude every speculative method with a verification contract.
Generic norm/GEMM fusion remains its nearest control.
No measured recurrence-specific benefit follows from this algebra alone.

## Candidate boundary

A potentially distinct recurrence-aware hybrid could combine stationary KV ownership with a physical-weight task frontier.
Its concurrent unit would be an already-ready operator task, not an uncomputed recurrent state.
The shared weights could change replication and grouping costs relative to an untied model.
The proposal still needs explicit messages and a costed resource schedule.
Ordinary CDB, TP, causal PP, and generic attention/MLP disaggregation are mandatory controls.

Guarded next-pass work is different from LoopSpec's future-token draft/verification work.
It uses a known recurrent input and speculates only on a continuation decision.
That distinction does not establish novelty over generic control-flow speculation.
If the candidate generates future tokens, LoopSpec and WaveFront become direct comparison requirements.

This contract establishes no new execution axis.
It identifies legal frontiers and state conditions that candidate mechanisms must satisfy.
It does not claim that no new axis can exist.

## CPU checks and next action

`scripts/check_exact_frontier.py` constructs 16 bounded graphs.
They cover L24/L48, one/three independent requests, fixed R4, and adaptive exits at R2/R3/R4.
Each request has two decode inputs.
Multi-request adaptive cases include different declared exit depths across requests and tokens.
Those exit traces define test graphs. They are not online knowledge supplied to a scheduler.

All 16 cases pass the ten declared dependency checks.
The checks cover norm boundaries, original hazard guards, sampling causality, local forks, KV providers, and independent-request edges.
The largest tested graph contains 24240 nodes and 30579 edges.
Its operation counts come from the checker, not a profiler.

The checker's two output reductions are symbolic dependency nodes.
Its owner labels distinguish roles and device indices. They do not simulate independent hardware lanes.
The checker does not schedule resources, test live gate values, or execute model kernels.
The source ledger, command, timestamp, and result summary appear in `artifacts/exact_frontier_contract.json`.

The cheapest implementation probe is mandatory-R2 prefix routing with unchanged original hazards and retained output semantics.
It should first check exits, logits, and terminal KV against the selected same-step path.
A performance result must separately report unchanged work, host-boundary savings, and any added buffering cost.
That probe addresses an exact adaptive optimization, not the full new-architecture objective.
