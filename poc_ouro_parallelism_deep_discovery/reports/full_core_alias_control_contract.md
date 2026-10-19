# Full-core alias counterfactual

Status: planned physical baseline control. This is not a new inference architecture.

## Question

Does repeated physical weight identity reduce latency under the actual full-stack reuse distance?
The earlier isolated-MLP control cannot answer that question alone.
Its intervening memory access is synthetic, not the native stack.

The new control retains the pretrained dense model, R4, BF16, and actual depth-indexed attention.
It compares aliased core weights with value-identical physical copies.
Both arms allocate every physical copy before timing.
This matches weight capacity and leaves pointer selection as the intended difference.

## Execution contract

1. Load the pinned original checkpoint into the ordinary physical layers.
2. Clone each physical block's parameters into three additional local copies.
3. Keep the original depth-indexed attention objects and KV ownership.
4. Build four execution lists with identical mathematical operations.
5. Use the original layer objects for all passes in the alias arm.
6. Use each pass's physical parameter copy in the materialized arm.
7. Reset residual state and apply the loop norm at the ordinary boundaries.

The copies include each block's normalization parameters.
Embeddings, final loop norm, gate declaration, and LM head remain shared in both arms.
The fixed native forward does not execute the adaptive gate.
The test therefore concerns fixed-R4 residency, not adaptive model quality.

## Required assertions

- Every copied parameter must equal its source BF16 tensor bitwise.
- Every materialized copy must use a distinct parameter storage pointer.
- Both arms must allocate the same set of block copies.
- The original core list must retain 24 or 48 checkpoint layer identities.
- Four registered parameter-block copies must remain distinct from that architectural identity count.
- The cache must retain 96 or 192 depth-indexed slots.
- Each pass must use the matching original attention object.
- Each complete forward must execute the original loop and residual order.
- Every timed request must produce its frozen teacher-token sequence.

A separate four-position capture compares logits, hidden state, and sampled KV.
Any numerical difference must remain visible.
Four-position agreement does not certify natural-generation parity or benchmark accuracy.

## Timing and memory

The native async scheduler, FlashAttention, and CUDA Graphs remain enabled.
The guard checks actual physical GPU UUIDs.
Each arm uses a new process and output directory.
Initialization and parameter cloning occur before the timed requests.
The report retains initialization costs separately where available.

The shape protocol remains 1K/4K/8K prompts and 128 controlled output tokens.
Each context has two frozen request shapes and five trials per shape.
The trial order is sequential across processes.
The test does not create a broad independent-request confidence interval.

Record unique parameter storage bytes, allocated/reserved HBM, and peak HBM.
Record cache slots and head ownership.
The generic module inspector will see four times as many attention wrappers.
Report that wrapper count separately from the original checkpoint layer identities and the shared KV slot count.
Do not call allocated weight capacity measured HBM traffic.
Do not infer a cache-hit rate without suitable hardware counters.

## Interpretation and stop conditions

A faster alias arm supports only a same-layout reuse effect at the measured operating point.
A negligible difference weakens that specific full-core residency premise.
Neither outcome proves a new distributed execution axis.
An alias-aware conventional unroll can use the same parameter identities.

Stop on wrong attention ownership, changed operation order, or unequal parameter values.
Stop if the four-copy setup exceeds the actual HBM limit.
Do not shrink precision or drop KV to make the setup fit.
Do not build a new engine to rescue this counterfactual.
