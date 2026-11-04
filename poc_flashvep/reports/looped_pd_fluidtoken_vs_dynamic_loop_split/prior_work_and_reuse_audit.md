# Prior work and execution reuse

Status: source audit before method experiments. No performance result is asserted here.

## FluidToken baseline contract

[FluidPD](https://arxiv.org/html/2610.06917v1) uses SGLang 0.5.4. FluidToken transfers prefix KV and executes the remaining suffix on D.
PPI compares cumulative predicted prefix service with remaining TTFT deadlines. DPI separately tracks active load and committed KV capacity.
The Decode predictor uses active batch size and aggregate context. Its available TPOT slack bounds the suffix budget.
The paper uses an adjustable slack multiplier. Values above one borrow request-average slack, not a strict all-token latency guarantee.
At each Prefill scheduling iteration, Algorithm 1 assigns budget to earlier queued requests to reduce predicted deadline misses.
Suffix time remains on the request's TTFT path. The algorithm does not remove model computation.
The inspected paper links and public search did not identify an official implementation. This is not proof that none exists.
Our implementation must therefore use the label `FluidToken-style reproduction` unless official code is independently verified.
The present runtime, BF16 geometry, models, and additional ITL gate differ from the paper's evaluation.
FluidRole is outside this campaign.

## Other relevant systems

[Load-Aware Prefill Deflection](https://arxiv.org/html/2607.02043v1) sends a request's Prefill to D and selects a TBT-safe chunk schedule.
This placement can avoid the inter-node KV handoff. It is not the same operation as transferring a computed prefix.
Its feasibility model accounts for active Decode and KV occupancy. That distinction matters when a low-utilization D cannot admit another long request.

[DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf) studies adaptive GPU multiplexing for Prefill and Decode.
Its motivation does not certify our recurrent models, runtime, workload, or SLO contract.
Chunked Prefill and existing Mixed execution are controls, not new contributions in this campaign.

## Existing exact connectors

`adaptive_loop_token_pd_2gpu/connector.py` already implements actual prefix/suffix migration.
It transfers full-depth prefix KV and applies the prefix frontier to the native scheduler.
Huginn also transfers the canonical initial state for the suffix. Regenerating that state independently would change the qualified execution.
The connector retains source ownership until commit and native acknowledgment.
Its dynamic rejection path resumes P work without discarding the prefix.

`adaptive_recurrence_cut/connector.py` provides the inherited recurrent-cut connector.
Reusing these facilities does not establish new method correctness at the current graph geometry.
Each new configuration still requires exact native-reference output and transferred-state tests.
The snapshot transport differs from Static PD's native streaming. A transport control is required.

## Reference engine

`ref_loopserve/README.md` documents a separate vLLM-based recurrent engine with depth-indexed KV and device allocation facilities.
Its `source/vllm/v1/core/kv_cache_manager.py` uses cache groups, allocation guards, and request release hooks.
These are not the current RLT connector's allocation or acknowledgment APIs.
The read-only reference is not a drop-in replacement for NIXL progress or exact migration.
Its early-exit and ragged-state paths are outside the fixed-R4/R32 contract.
Its selected fixed-row projection idea already informs the qualified numerical fixes. This campaign retains those fixes.

The reference records Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8, and a custom compiled vLLM payload.
Our qualified environment uses PyTorch 2.11.0+cu129 and CUDA 12.9. Binary and graph interfaces are not interchangeable.
The reference source retains an Apache 2.0 license. No new reference code was copied for this audit.

The reservation study distinguishes active allocations from promised completion credits.
Its statistical reservation policy depends on natural stopping and safe preemption. Our controlled output lengths provide no equivalent early-completion savings.
The device allocator changes block ownership and requires an append ring and lifecycle protocol.
Its recorded launcher excludes Huginn's auxiliary Prelude/Coda shadow path from that allocator.
These mechanisms therefore cannot safely replace the current R32 reservation path through a configuration flag.
The reference's copy thread waits for copy completion before the owner resumes a request.
That ownership rule is useful, but it does not establish a fix for the current NIXL/CUPTI interaction.

Audit sources: `README.md`, `research/tasks/2026-10-15_kv-reservation/README.md`, and
`research/tasks/2026-10-17_device-kv-allocation/README.md` under `ref_loopserve`.

## Novelty constraint

An unrolled logical-layer boundary can represent a recurrent loop cut.
At the same boundary, with the same state and cost, the mechanisms are equivalent.
Any claimed benefit must separate execution granularity, state movement, Decode interference, and online selection.
Neither an extra GPU nor a renamed logical-layer cut establishes Loop-specific novelty.
