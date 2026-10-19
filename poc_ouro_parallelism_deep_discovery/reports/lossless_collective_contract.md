# C08 lossless transport and collective contract

## Outcome

XOR can preserve every BF16 payload bit when both endpoints retain the same base bits.
XOR does not preserve floating-point SUM as an operation on encoded words.
BF16 arithmetic differences do not generally preserve the original values or the native reduction result.

Native TP gives every rank a reduced output.
It does not give every rank each peer's previous local partial.
A decoder that needs those peer-local bases requires additional state and an explicit ownership protocol.

Decoding each local operand before the unchanged native All-Reduce preserves its input bits.
That arrangement does not compress traffic inside the native collective.
A wire-coded collective needs a separate proof for its bases, arithmetic order, accumulator precision, and casts.
This audit does not establish such a collective.

## Scope and evidence classes

This audit uses installed vLLM source and seven deterministic CPU examples.
It does not execute a GPU kernel, a collective, a model, or an entropy codec.
The arithmetic examples refute unconditional equivalence claims.
They do not identify the installed GPU kernel's internal accumulation order or precision.

The source root is `/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm`.
The runtime is vLLM `0.26.0+cu129` with Torch `2.11.0+cu129`.
The JSON artifact records complete paths, hashes, and line locations.

| Source | Inspected location | Source-backed fact |
|---|---|---|
| `model_executor/models/ouro.py` | 78–91, 147–162 | MLP down projection and attention output projection use row-parallel linear layers. |
| `model_executor/layers/linear.py` | `RowParallelLinear.forward`, 1676–1702 | Each rank computes a local partial, then calls TP All-Reduce. |
| `distributed/communication_op.py` | 12–14 | The TP helper calls its group coordinator. |
| `distributed/parallel_state.py` | 641–668 | The coordinator selects an out-of-place collective operation. |
| `distributed/device_communicators/cuda_communicator.py` | `all_reduce`, 273–339 | Shape, configuration, and hardware can select different collective backends. |
| `distributed/device_communicators/pynccl.py` | `all_reduce`, 166–197 | The NCCL call receives tensor count, tensor dtype, SUM, communicator, and current stream. |
| `distributed/device_communicators/custom_all_reduce.py` | 230–280 | The custom path uses registered buffers and native tensor operands. It has no versioned decoder-base interface. |
| `model_executor/models/ouro.py` | 260–281, 367–374 | Hidden state and residual follow distinct normalization points. The loop reuses modules, not an activation-history cache. |

These paths do not establish which backend a later H100 run selects.
The custom path requires input size divisibility by 16, suitable layout, and topology-dependent eligibility.
Graph capture also changes buffer registration behavior.
Neither condition supplies a codec or a base cache.

## Byte-exact transport

Let `w(x)` be the exact 16-bit word of a BF16 value.
For a named base `b`, define `d = w(x) XOR w(b)`.
The decoder reconstructs `w(x) = d XOR w(b)`.
This identity also preserves signed zero, infinity, subnormal words, and NaN payload words.
It requires the same base bits, not merely numerically equal base values.

XOR alone uses the original number of bytes.
An entropy codec can reduce some payloads, but it adds processing, lengths, metadata, and buffers.
No ratio or processing time follows from the identity.
Any shared bit pattern can serve as a base.
Recurrence may affect compression statistics, but recurrence is not necessary for correctness.

Each message needs these identifiers:

- Model and runtime revision, dtype, shape, shard, and layout.
- Request generation, token positions, branch lineage, and row identities.
- Physical operator, physical layer, recurrent depth, and normalization or residual boundary.
- Sender identity, payload version, base version, and base owner.
- Encoded length, decoded length, and completion dependency.

The following protocol is sufficient for a point-to-point payload.
It is not an implemented codec.

1. The sender and receiver establish the same immutable base bits.
2. Each endpoint retains a lease on that base version.
3. The sender produces the complete current payload.
4. The sender encodes against the named base and sends the versioned packet.
5. The receiver checks the base identity and packet layout before decoding.
6. The receiver reconstructs the payload before its arithmetic consumer starts.
7. The receiver acknowledges completion after all decoder readers finish.
8. Each endpoint releases the base only after all dependent packets finish.

A missing base requires a full bootstrap, a charged fetch, or valid recomputation.
A version mismatch must not silently select another base.
Promotion to a new base requires explicit ordering between endpoints.
Branch rollback does not cancel a decoder reader that still holds a lease.

## Base ownership by layout

| Layout and proposed base | Who initially has the bits? | Additional contract |
|---|---|---|
| Native TP, sender's previous local partial | The producing rank, while its local buffer remains valid. | Retain that partial. Distribute or otherwise establish it at each decoder that needs it. |
| Native TP, previous global reduced output | Every participating rank receives that output. | Retain the exact prior output for each operator and row. Do not confuse it with a peer-local partial. |
| Remote MLP, previous MLP input | Both endpoints can observe it after its original transfer. | Retain matching versions at both endpoints. Placement changes and new workers need bootstrap. |
| Remote MLP, previous MLP output | The MLP owner and return receiver can observe it. | Retain matching versions on the same side of residual and normalization operations. |
| Replica-owned branch, borrowed immutable prefix | A replica may hold matching committed state. | A KV prefix does not imply the presence of the required hidden activation or projection output. |
| Replica-owned branch, private append | Its producer owns it. Other replicas need an explicit transfer. | Preserve lineage and leases during winner selection, loser rollback, and migration. |

Native buffers are transient operands, not guaranteed historical caches.
The all-reduced result is a global vector, not a collection of peer-local partials.
Custom IPC registration makes addresses accessible to a kernel.
Registration does not establish historical versions, retention, or a Python-visible collection of peer histories.

A previous global output is a legal common XOR base if all endpoints retain its exact bits.
That choice does not establish useful compression for local partials.
It also does not justify adding one global base to an arithmetic sum of local differences.

## State and wire accounting

Both audited Base sizes use hidden width `H=2048` and four recurrent passes.
They use `L=24` or `L=48` physical layers.
Each row-parallel output contains `M*H` BF16 values per rank before reduction.
Here, `M` counts current activation rows, not the number of cached context tokens.

One full payload uses `2*M*H = 4096*M` bytes.
A retained version for both row-parallel operators in every physical layer uses `2*L*M*H*2` bytes.
This accounting covers one state family: local partials or global outputs.
It excludes current operands, codec scratch, metadata, graph buffers, and KV.

| Retained version per rank | Base 1.4B, L24 | Base 2.6B, L48 |
|---|---:|---:|
| One family, M=1 | 196608 bytes, 0.1875 MiB | 393216 bytes, 0.375 MiB |
| One family, M=8192 | 1.5 GiB | 3 GiB |
| Both local and global families, M=8192 | 3 GiB | 6 GiB |

These are example structural capacities, not measured allocations or universal minimums.
Caching every sender's partial at every rank can multiply the local-partial family by `P`.
A chunked protocol may use another retention pattern.
Its proof must include every chunk's base owner and release point.

The first recurrent pass has no earlier pass for that token and operator.
Later passes can use only computed, retained versions.
Adaptive exits change active rows and can remove later-depth versions.
The packet must identify rows rather than assume a stable batch position.
No decoder can use an uncomputed next-pass value or a future gate result.

Wire cost includes encoded bytes, headers, length coordination, and any base transfer.
The state cache also incurs write and read traffic.
Any overlap claim needs ready inputs and the actual execution DAG.
This audit assigns no zero-cost bootstrap and no compression benefit inside an unchanged native collective.

## Arithmetic delta All-Reduce is a different operation

Let `p_i` be rank `i`'s current local partial and `b_i` its prior local partial.
Real arithmetic permits `SUM(p_i) = SUM(b_i) + SUM(p_i-b_i)`.
BF16 subtraction and addition do not generally preserve this identity bitwise.
The prior global output can already contain rounding that no local delta restores.

If every rank instead uses one common base `b`, real arithmetic requires `P*b + SUM(p_i-b)`.
One copy of `b` undercounts the base by `P-1` copies.
Correct base multiplicity still does not remove floating-point rounding differences.

Compressed words are not ordinary BF16 SUM operands.
Integer SUM, byte SUM, and XOR each define different operations.
Neither integer SUM nor XOR on difference words implements floating-point SUM of decoded values.
Reinterpreting encoded words as BF16 can create infinity or NaN unrelated to the source payload.
Entropy-coded streams also need framing and can have unequal lengths.
The installed NCCL call expects matching typed operands, not variable-length decoder packets.

## Decode placement and native order

| Proposed arrangement | Exactness condition | Native wire saving |
|---|---|---|
| Decode each rank's own payload, then call unchanged native All-Reduce | Restore identical operand bits. Keep backend, dtype, shape, participant order, and execution conditions unchanged. | None from this transformation. |
| Gather encoded peer operands, decode, then reduce locally | Establish every base and reproduce the required arithmetic structure. | Not established. The arrangement changes the collective algorithm. |
| Decode packets within a custom collective | Decode before each arithmetic combine. Match native stages, chunks, accumulation precision, casts, and operand order. | Conditional on a new protocol and measured costs. |
| Reduce BF16 arithmetic deltas, then add a prior global output | Needs a separate numerical proof. Local reconstruction alone is insufficient. | Not a lossless substitute by default. |
| Apply SUM directly to XOR or entropy-coded bytes | No ordinary BF16 SUM interpretation exists. | Invalid substitute. |

Byte restoration does not automatically preserve the native reduction tree.
The installed source interfaces do not reveal every binary kernel's internal tree or accumulator precision.
A staged wire codec can also require bases for intermediate partial aggregates, not only original rank operands.
Those bases need chunk, stage, participant, and version identities.
This is a protocol requirement, not a universal impossibility claim.

## CPU checks

The script executes only CPU tensor operations and integer identities.
All seven cases passed with `CUDA_VISIBLE_DEVICES=''`.

| Case | Observed result | Meaning |
|---|---|---|
| Wrong XOR base | Payload 1.5 decodes as 0.75 with base 0.5 instead of 1.0. | Base identity is necessary. |
| Special BF16 words | Five words reconstruct exactly, including signed zero and a NaN payload. | XOR preserves words, not only numeric values. |
| SUM of encoded words | Integer-code SUM then XOR gives 3.0. BF16-code SUM then XOR gives 2.0. Desired SUM is 3.5. | Encoded words are not SUM operands. |
| Rounded BF16 delta | Base 1024, current 1, delta -1024, reconstruction 0. | BF16 subtraction is not generally lossless. |
| Exact local deltas | Prior partials [256,1], current [256,2], deltas [0,1]. Prior SUM plus delta SUM gives 256, versus direct 258. | A rounded global base can invalidate reuse despite exact local reconstruction. |
| Different grouping | A BF16 left fold gives 0. A regrouped expression gives 1. | Arithmetic structure matters. Neither expression claims to be the native GPU kernel. |
| Common base multiplicity | One base plus delta SUM gives 2.5. Two bases plus delta SUM gives the required 3.5. | A shared base needs the correct rank multiplicity. |

The first script run used an incorrect expected value for the BF16-code example.
The corrected expectation is 2.0, and the complete script then passed.
This correction changed the test expectation, not a transport implementation.

## C08 decision boundary

An exact XOR wrapper is a generic transport representation with an explicit base cache.
Repeated physical operators may provide useful input statistics.
The same representation can apply to an equivalent logical unroll or a nonrecurrent stream.
It does not expose independent future recurrent work.

The existing C08 card retains `NO_NET_HEADROOM` for its tested unfused small-message premise.
Its prior CPU size ratios do not measure GPU codec time.
Its prior reversible-transform probe excludes entropy processing and receiver-base management.
This audit adds correctness and ownership constraints, not a new measured speedup.

Lossless decoding can preserve original-gate inputs when subsequent arithmetic remains unchanged.
A numerically changed reduction can change hidden states and later gate decisions.
The R4 and adaptive modes therefore need separate validation after any collective change.
The audit does not establish a new execution axis or a lossless arithmetic-delta baseline.

## Source and time record

Own segment start: `2026-10-19 11:44:46 UTC`.
Recorded end after checks: `2026-10-19 11:54:08 UTC`.
The recorded segment lasts 562 seconds.
The segment covers targeted source reads, CPU examples, report writing, and report checks.
It does not include a GPU measurement or another agent's work.

The report linter found zero violations.
JSON parsing, seven source hashes, the script hash, and capacity arithmetic checks passed.

Artifacts: `artifacts/lossless_collective_contract.json` and `scripts/check_lossless_collective_contract_cpu.py`.
