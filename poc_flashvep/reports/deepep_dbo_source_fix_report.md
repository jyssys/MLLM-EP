# Stock vLLM 0.20 DeepEP DBO source-fix validation

## Final gate

**FINAL STATUS: GO**

The source-level fixes reproduce the required greedy one-token correctness
recovery without the runtime monkeypatch. Attention-only restores every tested
text case, both fixes restore every tested vision case, both DP ranks pass, and
the required unequal-split, mixed-length, request-identity, and exception
lifetime checks pass. The optional heterogeneous text+vision wave and strict
long-form parity remain limitations described below.

This work adds no FlashVEP scheduling or performance optimization.

## Environment and method

- Model: Qwen3-VL-30B-A3B-Instruct, BF16
- Parallelism: TP2 / DP2 / EP4 / PP1
- All-to-all: DeepEP high-throughput
- GPUs exposed to the runs: physical 4,5,6,7 only
- vLLM: `0.20.0+cu129`
- PyTorch / CUDA / NCCL / DeepEP: `2.11.0+cu129` / `12.9` / `2.28.9` / `1.2.1+73b6ea4`
- Installed source patched for validation:
  `/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm`
- Runtime monkeypatch: unset for every source-fix ablation, regression, and
  latency run

The required matrix used greedy one-token generation, global requests 2/4/8,
three measured repetitions, and both DP ranks. Clean latency used two warmups
and seven measured iterations with tracing disabled.

## Root-cause decisions

### Attention metadata: confirmed

`GPUModelRunner._build_attention_metadata()` keyed reusable metadata only by
KV-cache specification and builder type. Two concurrent DBO ubatches could
therefore reuse one metadata object while `update_block_table()` refreshed only
part of its contents. The minimal fix adds `ubid` to the cache key and retains
the update-block-table optimization inside a single ubatch.

Causal evidence is direct: no-fix text fails at request 2, attention-only makes
text request 2/4/8 pass on both ranks, and DeepStack-only leaves the text
failures unchanged. With both fixes, the 399/400-token vision slices have
different FA3 scheduler metadata pointers on each rank:

- DP0: ubatch 0 `140649195709952`; ubatch 1 `140649195710464`
- DP1: ubatch 0 `139914888880128`; ubatch 1 `139914888880640`

### Qwen3-VL DeepStack: confirmed

The shared model-side DeepStack buffer was not sliced with the ordinary DBO
inputs. The source fix transports the real `UBatchSlice.token_slice` in the
existing forward context, consumes `buffer[start:stop]`, and retires the wave
only after all ubatch worker threads join. It has no `ubatch_id == 0`, suffix
offset, or exactly-two-ubatches assumption. Per-ubatch clear is suppressed;
the next payload write clears only a stale tail, outside active consumers.

Attention-only still produces wrong vision tokens (`198` instead of `1986` in
request 2/4), while both fixes produce `1986` everywhere. DeepStack-only vision
reaches the independent attention defect and raises FA3
`batch_size must be equal to batch_size_k`; the resulting DeepEP rank timeout
was stopped after the causal failure was captured.

## 2x2 causal ablation

Every PASS/FAIL below applies to global requests 2, 4, and 8 on DP rank 0 and 1.

| Source mode | Text | Vision | Observation |
|---|---:|---:|---|
| no fix | FAIL | FAIL | Text first fails at request 2 (`151645`, expected `2132`); vision first fails at request 2 with unstable wrong tokens. |
| attention only | PASS | FAIL | Text is exactly `2132`; vision request 2/4 is `198`, expected `1986`. |
| DeepStack only | FAIL | FAIL | Text matches no-fix; vision raises the FA3 batch-size mismatch before a token result. |
| both fixes | PASS | PASS | Text is exactly `2132`, vision exactly `1986`, all counts/repetitions/ranks. |

This is the predicted independent-defect pattern; results were not rewritten to
fit the prediction.

## Additional correctness regression

### Unequal token split and attention shapes

The 799-token vision prefill is split using the actual slices `[0:399]` and
`[399:799]` on both ranks.

| ubatch | DeepStack slice | Q shape | K/V shape | actual | query start | sequence length |
|---:|---|---|---|---:|---|---|
| 0 | `[0:399]` | `[399,16,128]` | `[399,2,128]` | 399 | `[0,399]` | `[399]` |
| 1 | `[399:799]` | `[400,16,128]` | `[400,2,128]` | 400 | `[0,400]` | `[799]` |

The scheduler metadata pointers differ as listed above. This passes.

### Mixed request lengths

A global request-4 wave placed 620- and 790-token text requests together on
each DP rank. DBO-off and DBO-on both return `2132` for every slot across three
measured repetitions. Submitted IDs such as `2-9469c3ba` / `3-9fb15acd` are
restored as engine IDs `2` / `3` in their original prompt slots. This passes.

### Distinct prompt and request identity

A bounded 16-token diagnostic used 790-token blue and red prompts. DBO-off
outputs become distinguishable at the second token (`5868` versus `4977`).
DBO-on preserves that distinguishing token and its request-ID association on
both ranks: blue remains request 0 with `5868`, red remains request 1 with
`4977`. There is no swap or output-list restoration error. This passes the
request ID to output identity check.

One red DBO-on continuation diverges from DBO-off after the identifying prefix.
The required one-token correctness remains exact, but this experiment does not
establish bit-exact long-form parity.

### Mixed text/vision wave

Unresolved, but optional under the spec. The private `_add_completion_requests`
harness stalls its input queue for more than 60 seconds in DBO-off before any
mixed-wave result. Both rank processes were interrupted and their
`KeyboardInterrupt` JSON was retained. Because it fails before DBO is enabled,
it is not attributed to either source fix.

### Exception and lifetime

Two focused tests pass:

1. actual slices `[0:2]`, `[2:4]`, `[4:6]` are consumed by three readers, then
   a shorter next wave clears the stale tail; this removes the two-ubatch
   assumption and checks stale payload handling;
2. a simulated consumer exception still invokes the wrapper finalizer exactly
   once, resets valid length, and leaves no completion state for the next wave.

The finalizer is outside the joined ubatch execution, so no active consumer is
cleared prematurely. It changes logical valid length only and performs no
asynchronous zero-fill on an ubatch stream.

## Clean post-fix latency

Values are the maximum of the two rank medians, in milliseconds. Speedup is
DBO-off / DBO-on, so values below 1 mean DBO-on is slower. Raw per-rank samples
are in the result directory.

| Modality | Global requests | DBO off | DBO on | Speedup |
|---|---:|---:|---:|---:|
| text | 2 | 2783.222 | 3465.742 | 0.803x |
| text | 4 | 2805.297 | 4695.291 | 0.597x |
| text | 8 | 2799.927 | 4035.365 | 0.694x |
| vision | 2 | 4371.737 | 4107.062 | 1.064x |
| vision | 4 | 4429.997 | 4709.892 | 0.941x |
| vision | 8 | 4496.938 | 4164.148 | 1.080x |

Correctness is recovered, but DBO is not a universal performance win in this
bounded eager-mode setup. No tuning was attempted.

## vLLM files and functions changed

- `vllm/v1/worker/gpu_model_runner.py`
  - `GPUModelRunner._build_attention_metadata`: include ubatch identity in the
    attention metadata cache key.
- `vllm/v1/worker/gpu_ubatch_wrapper.py`
  - ubatch forward-context construction: carry `UBatchSlice.token_slice`;
  - `UBatchWrapper.__call__` plus narrow lifetime helpers: finalize after all
    ubatch consumers, including exceptions and capture/replay paths.
- `vllm/model_executor/models/qwen3_vl.py`
  - `_get_deepstack_input_embeds`, `_set_deepstack_input_embeds`,
    `_clear_deepstack_input_embeds`, `_finalize_ubatch_inputs`: actual slice
    views and safe wave lifetime.

Reproducible patches:

- `poc_flashvep/deepep_revalidation/patches/vllm_0_20_dbo_attention_cache_ubatch.patch`
- `poc_flashvep/deepep_revalidation/patches/vllm_0_20_qwen3vl_deepstack_ubatch.patch`

Use `poc_flashvep/deepep_revalidation/set_vllm_source_fix_mode.sh` with
`none`, `attention`, `deepstack`, or `both` to apply/revert the validation
patches against the installed vLLM tree. The script compiles the touched source
and prints SHA-256 hashes.

## Artifacts and limitations

- Raw result directory:
  `poc_flashvep/deepep_revalidation/results/dbo_source_fix_20260807_170000/`
- Compact gate:
  `poc_flashvep/deepep_revalidation/results/dbo_source_fix_20260807_170000/summary.json`
- Report:
  `poc_flashvep/reports/deepep_dbo_source_fix_report.md`

Remaining limitations:

1. heterogeneous text+vision batching is unresolved in this private harness;
2. strict multi-token DBO-off/on equality is not claimed after the observed red
   continuation divergence;
3. the patch was validated against this exact vLLM 0.20 installation and is
   not presented as an accepted upstream design;
4. eager execution and this bounded workload do not characterize production
   DBO throughput.

## One recommended next action

Add an upstream-style vLLM test that submits a heterogeneous text+vision batch
through a supported engine API and asserts request-ID mapping plus multi-token
DBO-off/on parity; do not tune performance until that test is resolved.
