# Evidence index and Phase 0 review answers

## Evidence boundary

Source inspection supports implementation facts and design constraints.
Historical measurements support only the earlier runtime and workloads.
No new GPU inference, bandwidth, accuracy, or serving measurement ran in this planning task.
Read-only GPU inventory showed active burners on GPUs4–7. No process was stopped.

## Pinned source anchors

### Native vLLM

Pin: `568afb3a13806beb53bb2e6bd518269357b237c0`, tag v0.26.0.

| ID | Source | Audited fact |
| --- | --- | --- |
| V1 | [OuroAttention](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/models/ouro.py#L105) | Native projections, RoPE, attention, and per-depth namespace |
| V2 | [OuroModel](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/models/ouro.py#L292) | Physical layer construction, dormant gate, four local loops, ignored incoming intermediate tensors |
| V3 | [OuroForCausalLM](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/models/ouro.py#L377) | No SupportsPP interface in this class |
| V4 | [RowParallelLinear](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/layers/linear.py#L1541) | Parallel-input default, output reduction, and bias handling |
| V5 | [make_layers](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/models/utils.py#L798) | Contiguous layer ownership helper |
| V6 | [Llama native PP example](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/models/llama.py#L397) | First, intermediate, and last stage handling |
| V7 | [Worker.execute_model](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_worker.py#L1085) | Previous-send waits, predecessor receive, nonlast send, final-rank constraint |
| V8 | [GPUModelRunner.execute_model](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py#L4111) | One outstanding execute state and native forward context |
| V9 | [Native model forward and logits path](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py#L4350) | Intermediate result versus logits/sampling lifecycle |
| V10 | [EngineCore batch queue](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/engine/core.py#L617) | Queued model and sample futures |
| V11 | [KV cache shape construction](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py#L7290) | Backend shapes, strides, and layer packing |
| V12 | [KV specification discovery](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py#L7718) | Allocation follows registered attention layers |
| V13 | [FlashAttention cache shape](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/attention/backends/flash_attn.py#L130) | Logical cache shape and physical stride options |
| V14 | [Native KV update](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/attention/backends/flash_attn.py#L1071) | Slot-mapped K/V scatter path |
| V15 | [Graph wrapper](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/compilation/cuda_graph.py#L233) | Runtime mode, batch descriptors, capture and replay |
| V16 | [PP send](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/distributed/parallel_state.py#L998), [PP receive](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/distributed/parallel_state.py#L1093) | Transport primitives. These do not establish an Ouro cyclic executor |

### RLT and CDB

| ID | Source | Audited fact |
| --- | --- | --- |
| R1 | [RLT recurrent core](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/models/ouro.py#L283) | One full core, normalized output, trained gate |
| R2 | [RLT exit test](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/scheduler.py#L756) | Exit decision uses remaining probability |
| R3 | [RLT gate readout](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/worker/model_runner.py#L1109) | Cast before sigmoid |
| R4 | [RLT repair](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/kv_cache_manager.py#L747) | Current-position propagation into skipped-depth planes |
| R5 | [RLT finalization bookkeeping](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/kv_cache_manager.py#L438) | Bookkeeping can follow enqueue, not physical completion |
| R6 | [RLT batched repair and events](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/worker/model_runner.py#L1278) | Private Triton layout, stream event, and slot lifetime |
| R7 | [RLT async policy restriction](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/engine/stage_engine.py#L102) | Async mode requires delayed, random-lookahead, or trace policy |
| R8 | [RLT notice](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/NOTICE) | Standalone scheduler/cache runtime and Ouro attribution |
| C1 | [CDB policy](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/exit_policy.py#L150) | Sigmoid before float conversion, separate CDF and survival |
| C2 | [CDB no-refill wave](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L800) | Survivor shrink, wave coda, and repair flush |
| C3 | [CDB notices](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/THIRD_PARTY_NOTICES.md) | Root license does not replace file-specific notices |

The CDB [paper record](https://arxiv.org/abs/2608.09444v2) describes local depth batching and further asynchronous mechanisms.
This plan uses only the selected original-gate policy initially. It does not reproduce all paper variants.
The [official Ouro site](https://ouro-llm.github.io/) states that native vLLM runs all four loops without adaptive exit.
[PyTorch numerical guidance](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html) explains why arithmetic order can change floating-point results.
That guidance does not excuse a specific unexplained implementation failure.

## Historical evidence and limitations

All historical paths below use commit `2de9f955fe4eafc16bfb9086af7be07e37acfa74`.
The local source tree is `/home/esjung/MLLM-EP-ouro-request-pp`.
The audited report files and `stage.py` matched the pinned Git content where checked.

| Source | Evidence preserved |
| --- | --- |
| [analysis.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/analysis.md) | Earlier NO-GO and bounded scope |
| [benchmark_summary.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/benchmark_summary.md) | Natural quality and replay results |
| [correctness.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/correctness.md) | Historical strict TP numerical FAIL and exploratory quality |
| [memory_communication.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/memory_communication.md) | Explicit software payload and owner-local cache |
| [dynamic_policy_contract.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/dynamic_policy_contract.md) | Canonical gate and no-refill policy |
| [serving_summary.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/serving_summary.md) | Service clocks and differing output totals |
| [replay_summary.csv](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/replay_summary.csv) | Five-repeat frozen-work controls |
| [profiling_analysis.md](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/reports/ouro_8arm_static_dynamic_discovery/profiling_analysis.md) | Real cross-request kernels and heavy observer limitations |
| [request_runtime.py](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/ouro_8arm/request_runtime.py#L153) | Repair ACK completes before head and successor publication |
| [stage.py](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/ouro_8arm/stage.py#L52) | Current-row export/import and valid-length update |
| [native_quality.py](https://github.com/jyssys/MLLM-EP/blob/2de9f955fe4eafc16bfb9086af7be07e37acfa74/poc_flashvep/ouro_8arm/native_quality.py#L19) | Former native configuration, not a client SLO harness |

### Historical measurements, not new engine results

| Metric | Preserved result | Limitation |
| --- | --- | --- |
| Static natural service seconds, S/T/L/O | 6074.192 / 6219.233 / 3707.747 / 3311.532 | Custom family, different scheduling, preparation excluded |
| Dynamic natural service seconds, S/T/L/O | 4373.494 / 4267.465 / 3326.628 / 3777.943 | Custom no-refill. Not the proposed native integration |
| Static identical-work C4 seconds, S/T/L/O | 2.602 / 2.794 / 0.921 / 1.159 | Frozen 32-token outputs, five repeats, preparation excluded |
| Dynamic identical-work C4 seconds, S/T/L/O | 2.171 / 2.306 / 0.944 / 1.465 | Recorded depth schedule, not online Dynamic |
| Native TP1 / TP4 service seconds | 948.674 / 780.024 | Output totals 162994 / 178672. Not equal work |
| Dynamic Loop PP remote exit-KV bytes | 37006934016 | Software payload, not wire bytes |
| Dynamic Loop PP receive count | 188227 | Historical protocol, not the future optimal call count |
| Dynamic total remote bytes, Loop / Layer | 39656782256 / 14014785968 | Includes counted explicit tensor payload only |
| Static quality, S/L/O then TP | GSM232/300 and MATH61/100. TP233/300 and61/100 | TP remained exploratory |
| Dynamic quality, S/L/O then TP | GSM232/300 and MATH59/100. TP231/300 and61/100 | Separate dynamic policy and architecture effects |
| Custom TP P512 max hidden relative L2 | 0.469359, limit0.03 | FAIL remains unchanged |
| Bounded operator checks | 192/192 passed | Not full-model qualification |
| Heavy host-service overhead | 24.9%–405.0% | Not valid headline latency |

The old CDB control also used different packing and backend behavior.
Its bounded comparisons do not prove full-generation bitwise equivalence.
Old Loop PP repair-host intervals are not pure P2P times or removable E2E percentages.

### Historical report hashes

| File | SHA256 |
| --- | --- |
| analysis.md | `63a0574917aa718971580c801bbbf31156c45efedb0f21f53dee6d28c2101248` |
| benchmark_summary.md | `0fd2bb24f0fd75a347c961b9c8e339ffcae70311b2fb969c03f35c18570e259b` |
| correctness.md | `7771a10436bf17f0a0c726e6bce9a3ffd915c32b51ac54b46fcbae5d0a6f7e3a` |
| memory_communication.md | `629dc4797375f46fc93d827c5e2330bad2f2b5d1e8a2c492a7cfabc69dde8c51` |
| dynamic_policy_contract.md | `070d490c8666db893ee84903739c7ce2d3ee699e3a978a4d0bc470601ad5ea5a` |
| stage.py | `30cd99bd3286bef8d36361c76efda6d12a704996f7934c896579d9fe7f3af12b` |
| request_runtime.py | `df8657a51a060301c059ce57090d955b296b8ae4834dbfae31a1347ae1346570` |
| inherited `poc_async_recurrent_pp/scoring.py` | `7226e350e01cfb36661377ea549c87834579524a81317f93811e98f7f8b39b35` |

## Direct answers to the ten planning review questions

1. **Expose one loop:** adapt `OuroModel`, `OuroAttention` ownership, `OuroForCausalLM`, runner microsteps, and graph scopes. Preserve native operators. See V1–V3, V8–V15.
2. **Shared PP plumbing:** reuse tensor transport and processes, but not an assumed cyclic worker path. PP2 must prove linear depth ownership and physical-layer reentry separately. See V7, V16.
3. **RLT portability:** recurrent state and current-row policy are portable concepts. Its allocator, packed buffers, kernels, runner, and scheduler are not drop-in native interfaces. See R1–R8.
4. **Gate and TP:** the gate exists but is unused. Audit its default parallel input. Use one reduced gate result, one bias, one decision, and identical survivor order. See V2, V4.
5. **96 namespaces:** native attention names encode `24*r+l`. TP splits heads, physical PP splits layer owners, and Loop PP splits depth owners. See V1, V11–V14.
6. **Last-exited:** use native slot mapping and row views. Copy the current position only. Match CDB sigmoid and accumulation, not the current RLT arithmetic. See C1, R2–R5, V14.
7. **Overlap legality:** token t+1 needs a sampled token and ready earlier-depth history, not every deeper repair. A strict repair-inclusive wave barrier blocks it. See the exact DAG in architecture.md.
8. **Required evidence:** receiver completion, first dependent read, source lifetime, actual GPU overlap, and clean paired client deltas. Host concurrency or software byte counts are insufficient.
9. **Smallest fair Static test:** native TP1/TP4, PP2 proof, then matched PP4 and contiguous control at C1/C4/C8. G2 stops expansion without repeatable benefit.
10. **Minimum patch sequence:** native harness, linear adapter, bounded cyclic proof, synchronous Dynamic, Dynamic PP, then deferred repair. Stop before any engine-scale rewrite.

## Planning audit limitations

This audit inspected code, metadata, old reports, and selected old implementation paths.
It did not prove runtime cache packing, graph capture compatibility, gate TP correctness, or remote readiness on the proposed engine.
No new bandwidth, latency, accuracy, or capacity number is available.
RLT and CDB GPU dependency compatibility remains TO_VERIFY in isolated environments.
The performance gate is a future experiment, not a promised result.

## Planning-package checks

The package contains all 11 required documents, including the CSV matrix.
The matrix has 21 unique experiment IDs. All Phase 1–6 execution rows remain NOT_RUN.
Local document links and 39 pinned source links passed file-existence and line-bound checks.
Those checks do not validate future runtime behavior or remote link availability.
The prose passed structural lint with zero hard findings after excluding context-dependent synonym checks.
Eighteen advisory findings remained at the final check. Technical terms such as correct, fixed, and repair have distinct meanings here.
The pre-existing staged diff remained unchanged before this package was staged.
