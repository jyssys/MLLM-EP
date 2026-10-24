# Prospective research contract

This study uses the complete supplied specification as its experiment contract.
The registration contains the specification hash and the execution pins.
The prior scheduling verdicts remain unchanged.
The paused native-vLLM Ouro Loop PP project remains paused.

## Fixed computation

Ouro uses pretrained Base 1.4B weights at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Huginn uses pretrained 3.5B weights at revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
Every token requires Ouro R4 or Huginn R16.
Huginn's checkpoint default is R32. The registered R16 selects inference depth without changing weights.
Both models use BF16, original attention, and depth-indexed original KV values.
No model training, approximation, quantization, or dynamic exit belongs to this study.

## Legal execution modes

Pure prompt groups execute consecutive recurrent passes on one immutable token span.
Native Mixed combines prompt and decode recurrent rows for exactly one pass.
Mixed rows can have different starting depths.
No descriptor advances an unrelated decode row through a prompt group's extra passes.
All consumers retain native producer-event dependencies.
Huginn also retains the fixed injection tensor and separate prelude/coda KV.

The primary token chunk width is C512.
Matched secondary widths are C256 and C1024, when registered.
C does not change online.
Ouro supports prospective G1/2/3/4. Huginn supports prospective G1/2/4/8/16.
Each group clips at the remaining recurrent depth.

## Controls and progression

The first screen compares Atomic, Split, grouped-only, stock Mixed, completion-fair Mixed, fixed hybrid, and event-only Mixed.
Each paired cell uses the same GPU, backend, chunk, request inputs, and memory budget.
All Huginn split and hybrid arms share the same neutral boundary adapter.
Ouro retains its stronger native async execution.
Huginn retains synchronous execution. Cross-model raw latency does not define a winner.

P2 must reproduce opposing advantageous actions before adaptive policy development.
A practical signal requires approximately 3% complete-token improvement or a material TTFT–TBT Pareto improvement.
Independent server starts must reproduce that signal.
A changed nominal group without a changed executed plan does not qualify.
Failure against tuned Mixed blocks method promotion, not preservation of negative evidence.

## Measurement and statistics

Delivered output timestamps define TTFT and ITL.
A recurrent completion is not an output-token completion.
Clean cohorts exclude tensor copies, CUDA event instrumentation, and heavy tracing.
Plan-observer cohorts form a separate evidence lane.
Kernel service sums do not define a non-overlapped E2E breakdown.

Process starts and request-source families define independent uncertainty blocks.
Repeated tokens and loops do not define independent samples.
Reported paired intervals resample process blocks.
Small repeated mechanism prompts cannot establish benchmark accuracy or broad population equivalence.

Natural generation permits EOS. A response cap remains a cap, not successful natural completion.
Qualified request attainment requires completion, TTFT compliance, and compliance of every delivered ITL.
Sustainable capacity requires at least 1,000 offered requests per policy/rate and three independent starts.
Queue stability and the registered 90% attainment threshold also apply.
Model-specific SLO thresholds must freeze before candidate test runs.

## Resource and evidence policy

Eight H100 devices run independent single-GPU workers.
GPU UUIDs, process owners, source hashes, attempts, and raw outputs remain in the evidence ledger.
Only verified owned burners can stop.
Existing results and unrelated staged changes remain intact.
DP1/2/4/8 denotes serving replication, not new model parallelism.
