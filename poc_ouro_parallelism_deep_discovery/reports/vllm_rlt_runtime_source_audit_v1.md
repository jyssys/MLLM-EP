# vllm-rlt runtime source audit v1

## Decision and scope

vllm-rlt supplies another public original-gate implementation. The inspected code does not supply a mode-matched adaptive TP2/TP4 implementation or measurement. Its distributed path assigns a full model to each prefill or decode worker. This distinction corrects public-code exclusivity, not the campaign's measured baseline status.

This audit pins revision `9d601ceaf9af63121c742df5f584467e23630a0b`, dated 2026-10-14 08:14:53 UTC. The requested ThinkFlowLab URL redirects to InferMatrixLab. The [commit](https://github.com/InferMatrixLab/vllm-rlt/commit/9d601ceaf9af63121c742df5f584467e23630a0b) and [source ledger](../artifacts/vllm_rlt_runtime_source_observations_v1.json) preserve the identity and file hashes.

Work started at 2026-10-19 18:06:55 UTC. The ledger records the final timestamp and elapsed interval. This was source inspection and static metadata checking only. No installation, repository import, model run, GPU operation, environment change, or protected-result access occurred.

## Gate and KV mechanism

The model loads checkpoint-compatible parameter names, including `model.early_exit_gate`. Strict loading checks all tensor names and shapes. Each recurrent call executes every physical layer, applies the loop-boundary norm, and evaluates that gate. The normalized hidden state becomes the next loop's input. The inspected original mode adds no learned predictor. [Model, lines 223–290 and 318–368](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/models/ouro.py#L223)

The synchronous engine resets survival at each decode input. It consumes each completed loop's hazard, including hazards before minimum depth. It exits when the cumulative probability reaches the threshold after minimum depth, or at maximum depth. Full-depth prefill supplies the first output. [Stage transitions, lines 400–460](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/engine/llm_engine.py#L400)

LAST_EXITED uses depth-indexed page tables. Batch descriptors accept a separate depth for each request row. Terminal finalization copies the last executed depth's current-token K/V into every skipped depth plane. It preserves the already-computed shallower entries. This is copying, not terminal-page aliasing. [Storage and descriptors, lines 176–186 and 601–655](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/core/kv_cache_manager.py#L601), [finalization, lines 908–933](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/core/kv_cache_manager.py#L908)

These source checks do not prove all attention kernels or allocator lifetimes correct. They establish the intended execution and state mechanism.

## Comparison with the campaign

The campaign configures threshold 0.5, minimum depth 2, maximum depth 4, LAST_EXITED, full-depth prefill, and immediate original-gate consumption. It disables asynchronous batching and synthetic replay. It checks gate module identity and checkpoint gate tensors. [Campaign loader](../runtime.py:28)

| Field | Inspected vllm-rlt behavior | Comparison limit |
| --- | --- | --- |
| Threshold and bounds | The request class accepts `exit_threshold=.5, min_loops=2, max_loops=4`. Default threshold is 1. | Explicit configuration is necessary. Default means fixed depth. |
| Original gate | `ExitConfig(mode="ouro")` is the default. | Same checkpoint head does not establish equal arithmetic. |
| Immediate decision | Synchronous ordinary `ouro` consumes the completed loop's signal. | The engine rejects async ordinary `ouro`. |
| Async policy | The engine accepts delayed Ouro, random lookahead, or trace, apart from separate fixed-depth speculation. | These do not establish the campaign's immediate adaptive policy. |
| KV | LAST_EXITED is default. SHARED is a separate layout. | SHARED changes history semantics and is not mode-matched. |
| Graphs and scheduling | Graphs are opt-in. Default scheduler uses refill and 128-token prefill chunks. | Campaign settings and graph coverage do not transfer automatically. |
| Precision | The loader accepts BF16. | Kernel, batch-shape, norm and gate differences still need comparison. |

Sources: [configuration, lines 10–106](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/config.py#L10), [request controls, lines 6–38](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/sampling_params.py#L6), [async admission, lines 95–103](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/engine/llm_engine.py#L95), [layout distinction](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/docs/cdb_runtime.md#L110).

One concrete arithmetic difference prevents a source-only parity claim. RLT computes `logits.float().sigmoid()`, then transfers scores to Python floats. It tracks survival and tests `1-survival`. CDB computes `sigmoid(signal).float().item()`, then separately accumulates CDF and survival. Cast order can matter for BF16 signals. Algebraic equivalence in real arithmetic does not prove equal finite-precision branch decisions. This audit measured no divergence. [RLT runner, lines 495 and 521–524](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/worker/model_runner.py#L495), [RLT decision](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/engine/llm_engine.py#L451), [pinned CDB policy](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/exit_policy.py#L150)

## Devices, ownership and Base2.6

The offline facade accepts one device and constructs one engine. Attention and MLP projections are full dense linear modules. The loader places complete checkpoint tensors on that device. The inspected facade/config/model path exposes no tensor-parallel degree, sharded projections, collective reduction, or TP process group. This is a bounded source conclusion, not a repository-wide absence proof. [Facade, lines 18–54](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/entrypoints/llm.py#L18), [projections, lines 152–196](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/models/ouro.py#L152), [loader, lines 334–368](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/models/ouro.py#L334)

PDEngine uses spawned processes, one for each configured role/device. Each worker loads a full model and owns its cache and hidden slots. P owns prompt KV and does not sample. D owns sampling. NIXL transfers valid prompt KV ranges and the final prompt hidden state. D waits for all expected transfer completions before activation. These are P/D replica pools, not TP ranks or depth-sharded verifier stages. [Spawn, lines 122–160](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/pd/engine.py#L122), [worker, lines 46–105 and 122–168](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/pd/worker.py#L46), [activation, lines 329–393](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/pd/worker.py#L329), [transport, lines 20–98](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/pd/transport.py#L20)

Current PD defaults select delayed Ouro and async execution. A caller can supply different execution and exit configurations, but default PD is not the campaign policy. No ordinary-gate PD measurement follows from those configuration hooks. [PD defaults, lines 91–105](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/pd/engine.py#L91)

Base2.6 is not excluded by a hardcoded 24-layer geometry. The configuration parser reads checkpoint dimensions, and the layer list uses `num_hidden_layers`. A static field-name check found no unsupported keys in either campaign-pinned official Base config. Their layer counts are 24 and 48. This did not instantiate the class or load weights. [Parser and layers](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/vllm_rlt/models/ouro.py#L109), [pinned Base2.6 config](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/config.json)

The support matrix explicitly marks Base2.6 as shared implementation, not separately qualified. Thinking2.6 integration and quality checks do not qualify Base2.6 adaptive inference. [Support matrix](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/README.md#L107), [recipe limits, lines 220–233](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/docs/recipes/ouro_thinking.md#L220)

## External timing and quality evidence

PR30 merged as `8897fbaeac0822ee11d75461c7cffa240f4ab759`. Its protocol uses Base1.4 BF16 on B300, LAST_EXITED, adaptive threshold 0.2, and minimum/maximum depths 2/4. It tests 1,024/4,096/16,384/65,409 input tokens and 128 greedy outputs with EOS ignored. E2E uses 2C requests with replacement and includes prefill and drain. Decode-only excludes prefill. Three-trial medians exclude HTTP, tokenization, loading and warmup. P2→P3 changes to delayed stopping. The PR reports token/depth discrepancies and does not establish exact BF16 equivalence. Its raw campaign files remain local, not attached. These are historical external measurements, not current-HEAD or campaign measurements. [PR30, Final protocol and stage definitions, Known limitations](https://github.com/InferMatrixLab/vllm-rlt/pull/30)

PR31 merged as `37575cff0493c2e82b5008807321064077f89d06`. It compares four replicas with 1P3D, 2P2D and 3P1D. It uses Base1.4 BF16, FA4 split=1, async graphs, and 512 ShareGPT requests per phase with 128 outputs. Arrivals are 4 or 8 requests/s. Initial and replay phases receive equal weight. Latency summaries average phase percentiles, rather than pooling requests. Some runs share a host with concurrent configurations, while others are isolated reruns. Features change together. The body does not specify threshold or minimum depth. This is fixed-load PD evidence, not TP or peak-capacity evidence. [PR31, Test Plan and Attribution and measurement limitations](https://github.com/InferMatrixLab/vllm-rlt/pull/31)

The Base1.4 quality guide uses a selected fixed-depth GSM8K87 regression fixture. It explicitly denies a held-out interpretation. Adaptive runs do not inherit its stored fixed-depth quality gate. Its example permits threshold 0.5 but uses minimum depth 1. Therefore that example does not qualify the campaign's minimum depth 2. [Quality protocol, lines 3–29 and 144–194](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/docs/accuracy.md#L144)

The README warns about incorrect async-preemption outputs and unresolved async-speculation cleanup failures. Those warnings are combination-specific. They do not establish failure of synchronous ordinary-gate execution. Historical documentation also contains stale deferred-feature statements. Current code and checkpoint-specific qualification take precedence. [Warnings](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/README.md#L68), [historical runtime document](https://github.com/InferMatrixLab/vllm-rlt/blob/9d601ceaf9af63121c742df5f584467e23630a0b/docs/cdb_runtime.md#L1)

## Effect on the dossier

Public original-gate code is not exclusive to CDB. The [new scope overlay](public_adaptive_runtime_scope_overlay_v1.md) corrects that wording without changing old reports. RLT strengthens the generic mixed-depth batching and depth-aware KV baseline family. Prefix caching and P/D ownership are also public mechanisms. None alone establishes a new campaign architecture.

The inspected source does not close adaptive TP2/TP4 coverage, Base2.6 qualification, or numerical parity. No reported speedup or quality score enters campaign results. The validated CDB/native environments remain unchanged. This bounded audit ends without a runtime adoption or implementation proposal.
