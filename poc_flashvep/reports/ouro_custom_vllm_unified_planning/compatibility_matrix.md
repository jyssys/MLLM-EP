# Compatibility and qualification matrix

`SOURCE_VERIFIED` means the pinned source provides the stated mechanism.
It does not mean this planning study ran that mechanism on GPUs.
`HISTORICAL_MEASURED` refers only to the immutable earlier study.
`TO_VERIFY` means a future test is required.

| Feature | vLLM 0.26.0 Ouro | Current pinned RLT / CDB | Unified-engine decision |
| --- | --- | --- | --- |
| Base Ouro-1.4B BF16 loading | Native support. Historical runs exist | Separate loaders exist | Keep native loader. Requalify checkpoint identity |
| Static R4 Single / TP4 | Native optimized implementation | RLT is not a native TP plugin | REUSE. New clean qualification is NOT_RUN |
| Original gate weights | Present but unused by native forward | Both have active policy paths | Add a Dynamic-only gate path |
| Original q=.5 cumulative hazard | Absent from native execution | CDB and RLT have numerical differences | Use pinned CDB arithmetic |
| True loop skipping | Absent from native R4 forward | Separate runtime implementations | Add depth-level execution, not post-hoc output selection |
| Full-R4 prefill | Native forward executes four loops | Reference implementations support full-depth setup | Preserve |
| Depth-indexed KV | Four Attention namespaces per physical layer | Private depth-aware caches | Reuse native namespaces and pages |
| Paged KV | Native | Different cache managers and layouts | Do not import private RLT block tables |
| Layer PP | Ouro lacks `SupportsPP` | No directly portable native Ouro PP path established | Cyclic implementation required. HIGH_RISK |
| Loop PP | No native Ouro stage split | No verified drop-in native Loop PP4 | Linear stage adapter. TO_VERIFY |
| Contiguous 96-logical-layer PP | Not exposed for Ouro | Separate control required | Same Static partition as Loop PP4 |
| Whole-model CUDA Graph | Native graph system | Different graph ownership | Preserve native control |
| Adaptive survivor graphs | No native Ouro dynamic path | Reference runtimes use their own batching | Native stage/depth graph adaptation required |
| Synchronous last-exited | Not required by Static native | Implemented in reference runtimes | New native row publication required |
| Remote last-exited | No native Dynamic PP | No compatible proof for target native PP | TO_VERIFY, including receive/scatter completion |
| Async repair overlap | Not provided for target | RLT has different async policy restrictions | Separate experiment, not a feature claim |
| `ouro_delayed` | Not native | RLT delays a gate decision by one loop | Exclude from primary |
| RLT async scheduling | Not a native extension | Current StageEngine requires delayed, random-lookahead, or trace exit mode | Do not enable it as original-policy async repair |
| Shared KV across depths | Not the target native full-KV policy | Optional reference policy | Exclude from primary last-exited |
| Continuous request batching | Native serving support | Different local schedulers | Reuse native frontend. Freeze admission policy per comparison |
| Loop-level refill | Not native Ouro | Implemented in separate schedulers | Conditional port after no-refill correctness |
| Dense TP All-to-All | Not a required Ouro mechanism | Not relevant | Do not fabricate EP/A2A costs |
| New attention kernel | Not required by this plan | Private kernels cannot imply portability | Avoid unless reuse demonstrably fails |

## Environment compatibility

| Component | Audit observation | Qualification boundary |
| --- | --- | --- |
| Installed native environment | Python 3.14.6, vLLM 0.26.0+cu129, Torch 2.11.0+cu129, Transformers 5.16.1 | Metadata verified. Fresh inference NOT_RUN |
| Installed NCCL package | `nvidia-nccl-cu12` 2.28.9 | Actual loaded library and transport TO_VERIFY |
| FA3 | Earlier native traces used FA3 | Standalone `flash-attn` distribution is absent. Inspect vLLM bundled backend selection in Phase 1 |
| Source/wheel relation | Installed Ouro source matches pinned v0.26.0 source SHA256 | Whole-wheel rebuild and all extension identities remain TO_VERIFY |
| RLT dependency declarations | Torch >=2.5, Transformers >=4.54, separate package | Declared ranges do not prove native environment compatibility |
| CDB dependency declarations | Python >=3.12, Torch2.11.0, Transformers5.9.0, flash-attn-3 3.0.0 | Do not install these pins over the native environment |
| GPU hardware | Four H100 80GB HBM3, NV18 links in topology inventory | Effective bandwidth and simultaneous compute/copy remain unmeasured here |
| Model configured context | 65536, RoPE theta1000000, no sliding window | Start at qualified 8192. No 64K capacity or accuracy claim |

## Allowed process layout

| Test | CUDA mask | Logical mapping |
| --- | --- | --- |
| Native Single | `4` | rank0 -> physical4 |
| Two-rank proof | `4,5` | rank0 ->4, rank1 ->5 |
| TP4 / PP4 | `4,5,6,7` | rank0/1/2/3 ->4/5/6/7 |

The current planning task launched no inference processes.
Existing Qwen burners remained on all four allowed GPUs.
Future tests must release only those recognized burners, after checking for unrelated work.
Read-only inventory of a GPU does not authorize computation on it.

## Eight-arm target status

| Mode | Single | TP4 | Physical-layer PP4 | Loop PP4 |
| --- | --- | --- | --- | --- |
| Static R4 | Historical native support, new qualification NOT_RUN | Historical exploratory native results, new qualification NOT_RUN | Native integration NOT_RUN | Native integration NOT_RUN |
| Dynamic R2–R4 | Native integration NOT_RUN | Native integration NOT_RUN | Native integration NOT_RUN | Native integration NOT_RUN |

No new arm receives a correctness or performance PASS in this planning study.
The earlier custom-runtime results remain historical evidence, not results from this proposed engine.
