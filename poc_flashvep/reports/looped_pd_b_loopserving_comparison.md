# LoopServing and RLT numerical execution audit

Status: COMPLETED_WITH_LIMITATIONS. This file records verified source facts and completed probes.

## Scope and preservation

The baseline repository commit is `fea20edef840ed357ee0493c2f0dd69668c66c1f`.
LoopServing is a read-only reference at `poc_flashvep/ref_loopserve`.
The installed RLT engine remains unchanged at `/home/esjung/external/looped-pd-abc-rlt`.
All diagnostic patches are process-local and opt-in.
The new evidence namespace is `reports/looped_pd_b_numerical_diagnosis/`.
No serving-capacity sweep belongs to this task.

## Reference evidence and its limits

The imported reference contains these task reports:

- `research/tasks/2026-10-03_nondeterminism/README.md`.
- `research/tasks/2026-10-03_down-rows-fix/README.md`.

Both reports identify batch-dependent Down GEMM arithmetic in their H100 runtime.
They report repeatable outputs after fixing Down geometry in several tested workloads.
They do not prove that every engine needs only that change.
The saturation test also reports residual failures with the all-graph-row variant.
The imported task directories contain their README files, but not the described task-local raw results and scripts.
Thus those historical numbers are reference-reported results, not experiments reproduced in this task.

The general `research/tools/gemm_row_invariance.py` exists in the checkout.
The historical `select_fixed_rows_for_capture` hook is absent from the current imported source.
The current `ouro_static_dense.py` sets `_native_capture_rows = capacity` and executes an explicit `torch.mm`.
The historical one-line patch cannot be applied literally to the current RLT code.

## Compatibility

| Item | Imported LoopServing | Current RLT diagnostic environment |
|---|---|---|
| PyTorch | Baseline manifest: `2.10.0+cu128` | Measured import: `2.11.0+cu129` |
| Python | Baseline manifest: `3.10.20` | Existing RLT environment, recorded by commands and source lock |
| CUDA toolkit used by Torch | Baseline manifest: CUDA12.8 build | Measured `torch.version.cuda`:12.9 |
| Ouro projection path | Merged/static dense, capture-stable `torch.mm` | Separate `nn.Linear` Q/K/V/O/Gate/Up/Down |
| Huginn projection path | vLLM parallel linear components | Native RLT `nn.Linear` components |
| Normalization | vLLM RMSNorm and optional fused kernels | PyTorch FP32 `square().mean()` and `rsqrt` |
| Attention | Reference reports FA3-based paths | Measured FA4 `4.0.0b30` |
| KV layout | Reference has policy-dependent pooled depth KV | Frozen last-exited, exact fixed-depth planes |
| Graph structure | Device-controlled elevator and fixed projection slabs | Recurrent graphs, boundary graphs, host scheduler |
| Source license | `source/LICENSE`: Apache2.0 | `LICENSE`: Apache2.0 |

No reference engine or kernel was copied into RLT.
The adaptation uses independently written projection padding and normalization diagnostics.
These are source-license observations, not a legal review of all third-party dependencies.

## Completed operator evidence

Ouro sample3 uses59 prompt tokens. Its historical first output difference is index125.
Teacher-forced replay preserves the exact prompt and generated history before that output.
The first observed difference for its selected prompt row occurs at loop0, physical layer0, `mlp.down_proj`.
The Down operands match exactly between prompt batch1 and2.
Nine of2048 output elements differ. Maximum absolute error is0.0001220703125.
Relative L2 error is3.881517841364257e-06.
This is an instrumented native replay, not proof of the historical PD run's earliest instruction.

Down-only geometry does not suffice for the tested teacher-forced histories.
After fixing Down, Ouro first differs at decode input step51, depth3, layer20, `post_attention_layernorm_2`.
Its norm input matches exactly. Eight of2048 output elements differ.
Maximum absolute error is0.001953125. Relative L2 error is0.0006544701172970235.

Huginn sample92 uses1614 prompt tokens. Its historical output difference is index71.
In the all-step replay, `core_block.2.norm_1` first differs at decode input step22 and depth14.
The input matches exactly. Twenty-eight of5280 output elements differ.
Maximum absolute error is0.015625. Relative L2 error is0.000576566148083657.
The later forced-history greedy output changes between batch1 and2.

The Huginn real-operand sub-probe isolates FP32 mean reduction.
Its variance differs by1.1920928955078125e-07 at M2/M3.
The normalized FP32 value differs by at most4.76837158203125e-07.
BF16 conversion increases the maximum difference to0.0078125 before the learned norm scale.
The scaled norm output differs by at most0.015625.
The small-row fixed-reduction candidate matches M16 for24 tested row counts from1 to256 on this operand.
That result is an operator regression, not a complete model-correctness acceptance.

## Instrumentation corrections

The first teacher-replay version reported `topk(...).indices[0]` as `top1`.
That field is not greedy argmax when logits tie.
The original raw files remain unchanged and are not accepted as greedy-output evidence.
Subsequent versions record `torch.argmax`, plus the separate top-k order.

Huginn calls `transformer.ln_f` twice in one coda pass.
A trace key without an occurrence index pairs the wrong calls.
That artificial difference is excluded. The corrected trace key includes the occurrence index.
The norm finding above uses `core_block.2.norm_1`, not the duplicate coda key.

## Qualification outcome

The selective Down-plus-norm candidate passes Ouro's tested200/400-request internal gates.
It includes the old failing seed and a new independent400-request seed.
Huginn passes the original200-request seed but fails199/200 on the independent seed.
Its400-request closed-loop baseline repeat also fails393/400 before B starts.
Thus the reference helped identify a real cause, but its Down-only result does not transfer unchanged to this runtime.
Fixing every Linear is neither justified nor accepted by this evidence.
Subsequent real-history probes identify Huginn's adapter as another susceptible projection.
Its selective correction passes the two minimal residual replays.
Its independent200/400-request qualification is still running.

See [the model-specific qualification](looped_pd_b_numerical_correctness.md).

## Remaining limits

Projection and normalization susceptibility do not exclude a B transfer defect.
Actual PD repeats, independent arrivals, prompt byte fingerprints, coda validity and full-depth work must also pass.
Original HF fidelity and downstream answer accuracy remain separate gates.
No lossless original-model equivalence is claimed.
