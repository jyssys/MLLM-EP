# Native implementation

Scope: `CONTROLLED_NATIVE_LP_EP4_HARNESS`.

The extension is isolated under this PoC's startup hook; installed vLLM source
and earlier PoCs are not edited. The extension adds request stage state to the
real scheduler. Before the last stage, it suppresses only the injected request's
sampling result, retains its encoder outputs, and leaves its prompt-computed
count pending. Decode requests keep normal scheduling and token sampling.

Each physical model execution traverses 48 decoder layers in order. Before and
after the active range, only resident decode rows execute attention/router/MoE.
Inside the range, the full injected prompt and resident decode rows execute
together. FlashAttention metadata uses the resident requests' original block
tables and KV slots outside the active range. DeepEP sees correct per-DP/SP row
sizes for each branch. All four EP ranks follow the same layer/collective order.

Prefill hidden and residual tensors are stored in a GPU dictionary indexed by
request ID. A later iteration consumes them before advancing its next contiguous
range. DeepStack remains injected at its original layers. The final stage alone
reaches usable prefill logits. Intermediate logits are sampled by the stock
runner but discarded by the scheduler; this prototype overhead is included in
clean timing. The normal model, sparse expert kernel, routing rule, BF16 dtype,
and paged attention are used throughout.

GPU row gathers/scatters and a small DP stage header are also included.
The final revision piggybacks the header on the installed runtime's mandatory
DP padding synchronization; it does not introduce another collective inside
the language-model forward. Earlier conditional CPU-header revisions raced
idle/dummy calls at wave transitions and are retained as failed diagnostics.
These are integration costs, not evidence of an intrinsic EP gap.
The ordinary Full/CP arms use the same hook and runtime configuration.

This is actual stateful model execution, not captured MoE replay or summed
timing. Serving is a bounded manually stepped two-DP harness; no production
queueing system is claimed.
