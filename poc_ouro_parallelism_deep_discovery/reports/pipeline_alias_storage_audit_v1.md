# Pipeline graph expression and physical parameter storage

## Result

The installed pipeline IR preserves the CPU toy's outputs, but it does not preserve active cross-stage parameter aliases.
The original toy stores 640 parameter bytes.
An explicit aliased unroll and an exported program retain those 640 bytes.
The default pipeline IR uses 2560 bytes for active parameter reads.
It also retains unused parameter registrations, giving 3200 registered bytes in this test.

These are CPU storage observations, not Ouro performance results.
No CUDA context, distributed worker, model checkpoint, KV cache, or original exit gate enters the toy.
The observations limit a baseline assumption.
They do not create a new candidate or reject every recurrence-specific architecture.

## Primary source and implementation pins

The live PiPPy HEAD is `1bcb2bfb2d6cc4ac2125c0edb37c35585bb9695f`.
GitHub reports its commit time as August 21, 2024, at 16:31:24 UTC.
The repository page reports archival on August 5, 2025.

The pinned [PiPPy architecture document](https://github.com/pytorch/PiPPy/blob/1bcb2bfb2d6cc4ac2125c0edb37c35585bb9695f/ARCHITECTURE.md) describes both shared-parameter policies.
TRANSMIT retains a source parameter and supplies its value to another stage.
REPLICATE gives each consuming stage its own parameter copy.
The document also describes stage dependencies and nonlocal edges.
These statements appear at raw lines 13,104–122.

The document's old example describes TRANSMIT as the default.
The pinned [PiPPy implementation](https://github.com/pytorch/PiPPy/blob/1bcb2bfb2d6cc4ac2125c0edb37c35585bb9695f/pippy/_IR.py#L1148-L1193) already selects REPLICATE in public tracing.
Its internal handling still contains TRANSMIT and REPLICATE branches at lines 909–1047.
The old example must not substitute for the pinned public API.

The tested installed package is PyTorch `2.11.0+cu129`.
Its recorded Git revision is `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
Its local `_IR.py` SHA-256 is `c4c47afeeeaa86aa6833a66a521ad737c5aec0f186715833e304f04099eb31f6`.
The raw official file at that revision has the same hash.

The installed [public tracing path](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/distributed/pipelining/_IR.py#L1054-L1079) selects REPLICATE.
The constructor explicitly deep-copies shared parameters at lines 588–605.
Its comment gives a training-gradient rationale.
The CPU inference toy still takes this copy path with `requires_grad=False`.

The installed public `pipeline` signature exposes no `multi_use_param_spec` option.
Its private `_from_traced` function retains that argument, but the inspected AST contains no reads of it.
An enum named TRANSMIT does not establish a usable installed public TRANSMIT mode.
This audit does not call that private argument or install archived PiPPy.

## Deterministic CPU control

The [script](../scripts/pipeline_alias_cpu_v1.py) creates four distinct nonlinear physical blocks.
Each block uses a fixed 4×4 matrix and a fixed four-element bias.
The function applies `tanh(linear(x)) + x/8`.
The core executes four times, giving 16 dependent block calls.
The input has two rows and four columns.
Every tensor uses CPU FP64.

The script uses no random initializer, optimizer, learned model, GPU backend, or process group.
It reads `torch.cuda.is_initialized()` before and after the test.
Both status flags are false.
The command also sets `CUDA_VISIBLE_DEVICES` to an empty string for that process.

Two split policies exercise the public pipeline conversion:

- The depth policy splits after each full core visit, giving four logical stages.
- The physical policy splits after each block call, giving 16 logical stages.

The second policy names a potential physical-layer ownership mapping.
Its default IR still contains 16 separate stage modules.
The policy name does not claim that the runtime co-locates or aliases those modules.

| Representation | Unique parameter storages | Parameter storage bytes | Output versus original |
|---|---:|---:|---|
| Original physical core | 8 | 640 | Reference |
| Explicit aliased unroll | 8 | 640 | Same module calls |
| Exported state | 8 | 640 | Bitwise equal FP64 |
| Default four-stage IR, active reads | 32 | 2560 | Bitwise equal FP64 |
| Default 16-stage IR, active reads | 32 | 2560 | Bitwise equal FP64 |
| Either default IR, all registrations | 40 | 3200 | Includes unused registrations |
| Manual local alias control, 16 stages | 8 | 640 | Bitwise equal FP64 |

Each default IR has 56 named parameter references when duplicate references remain visible.
Its active graph reads use 32 distinct parameter objects and storages.
The active storage sets do not intersect the original core's storage set.
The recorded maximum output error is 0.0 for both split policies.

The raw artifact records Python object IDs, storage pointers, offsets, shapes, strides, and graph text.
These addresses identify storage only within the recorded process.
They are not reproducible address values across runs.
The [raw metadata](../artifacts/pipeline_alias_cpu_v1.json) contains the exact package and script hashes.

An initial assertion assumed that all registered bytes equaled four active copies.
That assertion failed.
Inspection found unused original parameter registrations beside the active copied sets.
The revised control records registered storage and reachable parameter reads separately.
The report does not silently replace that unexpected observation with the original expectation.

## What the manual control does

The final toy control explicitly rebinds corresponding physical parameters inside the 16-stage IR.
It also rebinds unused registrations.
All 16 stage modules remain on CPU in one process.
The rebinding restores eight unique storages and the original output.

Conceptual owners are the four physical block indices.
Repeated visits to one block would belong to that block's owner.
This construction demonstrates a possible local representation with preserved aliases.
It is not default pipeline conversion.
It does not validate cross-process pointer sharing, device movement, scheduling, or communication.
It changes only newly created toy objects, not installed source.

## Physical-layer placement versus logical-depth PP

Let W denote the unique BF16 dense matrices in the physical core.
The seven matrices per layer give W=2352 MiB for Base 1.4B and W=4704 MiB for Base 2.6B.
This definition excludes norms, embeddings, LM head, gate, KV, scratch, and allocator overhead.
The following values are structural accounting, not toy or GPU measurements.

Assume balanced contiguous partitions, R4, and P=2/4/8.
Logical-depth PP partitions the four logical core visits into consecutive stages.
One stage can reuse a physical parameter within that stage.
Different device owners still need local copies or a charged alternative access mechanism.

| P | Logical-depth PP aggregate dense weights | Physical-layer cyclic aggregate dense weights | Native TP aggregate dense weights |
|---:|---:|---:|---:|
| 2 | 2W | W | W |
| 4 | 4W | W | W |
| 8 | 4W | W | W |

At P2, each logical-depth stage contains two complete passes and can reuse one full physical core.
At P4, each logical-depth stage owns one complete pass.
At P8, each stage owns half a pass, repeated across four depth groups.
These equations describe active required matrices, not the toy's extra unused registrations.
They do not predict installed distributed stage allocations.

Physical-layer cyclic placement instead assigns each physical layer partition to one owner across all four passes.
Each rank retains approximately W/P dense bytes.
The activation visits the physical partitions in order, then returns for the next pass.
This ownership can remove replicated weights relative to logical-depth placement.
It cannot improve aggregate dense-weight capacity relative to native TP, which already stores one aggregate physical copy.

The native per-rank dense-core counts are 1176/588/294 MiB for 1.4B at P2/P4/P8.
They are 2352/1176/588 MiB for 2.6B.
Native TP partitions each matrix rather than assigning complete physical layers.
Both layouts can reach W aggregate dense bytes under their stated allocation rules.
This comparison does not call every model tensor minimally replicated.

TRANSMIT is a different ownership choice, not a free global alias.
A remote stage needs the parameter value through transfer, retained local storage, or remote reads.
Retaining an immutable received value can amortize transfer, but it creates a local copy.
Leaving a parameter on its source requires charging the consumer's access path.
This CPU audit measures neither mechanism.

## Useful work and communication do not follow from storage equality

For balanced cyclic physical partitions, a fixed-R4 token crosses `4P−1` core boundaries.
This count includes three loop returns and excludes embedding/head placement.
It gives 7/15/31 boundaries at P2/P4/P8.
Each simple boundary carries a complete current hidden row, or a batch of such rows.

Native TP instead requires the relevant partial-output reductions throughout each physical pass.
Counting collectives and activation boundaries does not compare their latency or bytes automatically.
Their kernels, participation, overlap, message payloads, and contention differ.
A physical partition also executes full local matrices instead of pooling P devices for each projection.

One request supplies one causal current-token stage at a time in this non-speculative control.
Four passes do not create four ready stage inputs.
For C independent requests, at most `min(P,C)` cyclic stage workers can have distinct request work simultaneously.
Actual alignment can make that number smaller.

| System C | P2 bound | P4 bound | P8 bound |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 2 | 2 |
| 4 | 2 | 4 | 4 |
| 8 | 2 | 4 | 8 |

These are concurrency bounds, not utilization or performance estimates.
Native TP can already use P cooperating devices on one ready row.
Prefill chunking or speculative tokens need separate causal and state contracts.
The toy supplies neither.

Original-gate adaptive execution also needs actual continuation decisions after each completed loop.
Physical weights can remain resident when a request exits early.
The runtime must still preserve selected hidden states and depth/layer KV identities.
The toy does not test `last_exited`, gate ordering, KV publication, or adaptive Base 2.6B support.

## Interpretation limits for this campaign

An aliased logical unroll proves that the same fixed-R4 function can retain shared parameter identities in a representation.
The successful CPU export strengthens that limited observation.
Neither statement proves that the installed pipeline conversion preserves those identities.
The toy directly shows that its tested default conversion does not preserve active aliases across stages.

A generic graph can also describe cyclic physical ownership after an implementation supplies the required mapping and state protocol.
Graph expression alone does not establish that an efficient implementation exists in the selected runtime.
Physical placement can therefore remain a legitimate empirical question.
It needs actual ownership, state, capacity, communication, and critical-path evidence.

A gain against four materialized depth copies would establish only that weak comparison.
A qualifying result must also compare with alias-aware controls and native TP's existing aggregate residency.
This audit supplies no speedup, publication novelty, or architecture-wide rejection.
It preserves the [alias-equivalence caution](alias_equivalence_interpretation.md).

## Reproduction and actual time

```text
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/cp-composition-vllm026/bin/python scripts/pipeline_alias_cpu_v1.py --output artifacts/pipeline_alias_cpu_fresh.json
```

The program refuses to overwrite an existing output.
It uses installed packages without editing them.
The source observation file records the failed initial expectation and the source pins.

Work began at 2026-10-19 14:12:45 UTC.
The final CPU test body ran from 14:16:17.825716 to 14:16:18.953326 UTC.
Those timestamps exclude interpreter and import startup.
The source observations record the segment end after report validation.
Concurrent worker intervals do not add campaign elapsed time.
