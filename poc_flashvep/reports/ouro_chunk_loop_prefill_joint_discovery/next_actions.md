# Completed study and optional next action

The active experiment is chunk/loop prefill scheduling, not vLLM Loop PP.
The branch is `flashvep/ouro-chunk-loop-prefill-joint-poc`.
The initial staged fingerprint is `7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9`.
Preserve the 68 unrelated staged entries.

`confirm.sh` completed corrected warm confirmation, diagnostic controls, and the 8K correctness probe.
Its EXIT trap restored the GPU4 burner. GPU5–7 burners remained active.
No GPU experiment remains running for this study.

Completed checks:

1. Analyzed 326 completed serving cells, including separately labeled calibration and diagnostic cells.
2. Completed 100 measured warm-confirmation cells across three starts and ten paired cohorts per configuration.
3. Checked 31 pretrained non-reference numerical comparisons and 1,551 full-R4 issued-work records.
4. Exported the physical Nsight kernel timeline. Preserved private binary artifacts on the server.
5. Marked Phase 3 and Phase 4 NOT_RUN because the registered continuation gate did not pass.
6. Created lossless raw archives and checked their SHA256 hashes.

The verdict is `CONDITIONAL_FIXED_QUANTUM_ONLY`.
Do not resume the paused PP program or automatically build Adaptive Joint.
The optional next experiment is a counterbalanced, fully warmed 8K/concurrency8 tail-SLO test with complete outputs.
It must compare useful quanta against tuned Atomic and stock Mixed before any new method claim.

The initial `fa4_confirm_s0` is an aborted, one-sided cold-JIT lane.
It is not headline evidence. The correction is documented in `failures.md`.
