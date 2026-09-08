# User-supplied Libra supplementary code

Received during ongoing three-baseline screening on 2026-09-07.
README.md and REPRODUCE.md read completely. Files are original, unmodified inputs.

| File | SHA256 |
|---|---|
| README.md | d401e254a6a5c0a1a738d20f26fbbfc28404cfb22568454a58777871092b59db |
| REPRODUCE.md | a974851957cd3afeee4d3986064539158b74849a67b24a527a19570a45e8c6f3 |
| sglang_libra.diff | 7775be51c3092eea5c74740f378acaa01530998161006ee00fa6e0e045277ee4 |
| sglang_libra_internal.diff | bef3a95ecd75f6878b1c3c57bdc81f9e50addebc6dfa7bc85aba466fa65a9b61 |
| sglang_lina.diff | a2c3f38e3dad98e4eb3106a6efe8ecb6fa0ecc2985468e956c78b72865e3cec5 |

Documented SGLang base: `023288645b80fb41b3eed55fd413dd69a7904593`.
Python 3.10, CUDA 12.6+ (reproduction hardware used CUDA12.8, 8 H200).
Libra main is throughput implementation; internal build is route/imbalance
measurement; Lina is the separate table-prediction predecessor.

Safety: scripts are not executed blindly. Main diff deletes many tests and
contains benchmark-specific paths/options; apply only in a fresh isolated clone.
GPU launches will explicitly restrict physical GPUs 1/2/3/4, regardless of script
defaults. No original user file or working serving installation is overwritten.

## Audit priorities

1. Compare actual Cython replication/sharding to Appendix B/C and prior fallback.
2. Verify current-input / next-gate predictor tensor boundary and layer mapping.
3. Trace actual local/remote overlap, copy-engine streams, double-buffer lifetime.
4. Inspect `is_ori`, N/L, scheduling, shape and hardware assumptions.
5. Separate throughput benchmark from imbalance postprocessing and from online
   request-level serving; port Qwen-VL only after the baseline mechanism is sound.

## Initial source/build results

- All three patches apply to the documented commit. Main patch's unrelated
  deleted `test/lang/example_image.png` lacks a full binary index; only that
  deletion is excluded. All implementation hunks apply unchanged.
- Isolated `/home/esjung/.venvs/libra-supplement-py310`: Python3.10.12,
  SGLang0.4.10, torch2.7.1+cu126, transformers4.54.1. Qwen source imports pass.
  Full dependency freeze is saved; the vLLM baseline is unaffected.
- Unmodified actual Cython planner compiles separately under Python3.12 for
  route analysis. 48 synthetic conservation/placement invariants PASS. These
  tests establish implementation safety, not a scientific synthetic finding.
- Actual Phase2 planning moves remote assignments immediately after duplicating
  an expert and searches alternative overloaded ranks/experts. This is stronger
  and more explicit than the provisional literal-PDF port. Earlier port counts
  are superseded by `analysis/libra_supplement_full96/`.
- Main and internal Cython algorithms differ only in clearing the output mask
  internally; main runtime clears it at the caller. Preserve that lifecycle.
- Source's primary path is AllGather / local / remote / AllReduce, not DeepEP
  HT. `forward_mid` predicts with next gate on current post-attention-normalized
  local hidden states. This agrees with the fresh HF predictor probe boundary.
- Important implementation assumptions: equal source sequence lengths,
  `SEQ_LENS_SUM` fixed metadata buffers, compile-time MAX_GPUS8 / MAX_EXPERTS160,
  N/L and maximum sending-peer constraint, original weights copied into even/odd
  large buffers every layer. These are candidates for measurement, not failures.
- Supplied throughput scripts hard-code GPUs0–7 and original8GPU topology. They
  will NOT be executed as-is. A harness uses only physical1–4. Reduced-layer
  dummy-weight runs are functional checks only, and concurrent MoDES activity
  makes their timings ineligible as performance evidence.

## Native functional check — 16:20 KST

`online/libra_native_dummy_sanity_v3` exits successfully on all four allowed
GPUs. Two measured randomized paired repetitions and one warmup use the
unchanged decoder's `is_ori` switch. All eight measured Libra rank outputs
match the corresponding vanilla logits exactly. This is a four-layer dummy
model: small dummy expert contributions may make the comparison insensitive.
It proves that the provided communication/buffer path executes, NOT that real
Qwen weights or a full MLLM port are numerically verified. No speedup claim is
drawn from these runs (concurrent calibration, compilation, reduced model).

Environment fixes were confined to the isolated environment: expose its Ninja
binary on PATH, and use PyArrow20 with the pinned datasets2.14.4 instead of
PyArrow25, which removed PyExtensionType. No assertion or model math was bypassed.
The regenerated dependency freeze includes this compatibility correction.
