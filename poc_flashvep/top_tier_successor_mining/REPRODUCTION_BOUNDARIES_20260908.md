# Reproduction boundaries before ranking

All methods receive algorithm,source,quality/functional and runtime screening.
These labels are deliberately not interchangeable with original-paper speedup
reproduction or cross-model validation.

| Method | Faithful components executed | Deployment changes | Not reproduced |
|---|---|---|---|
| SERE | Official CUDA rerouter;400×128 official BF16-norm similarity table;unchanged weights;S/rho controls | Qwen-VL model and vLLM.20/DeepEP port;DP-local batch union reconstructed across TP sequence shards | Original single-H20/vLLM.8.4 performance;EP-global union;Kimi |
| Libra | Supplied native SGLang decoder,actual next gate,Cython planner,replication copies,local/remote split and collectives | Four H100,30B model;captured VL embeddings/MRoPE/DeepStack bridge;bounded equal-length real-source groups | 8H200235B/355B headline;full continuous VL serving/vision timing;full generation task quality;Kimi |
| MoDES | Official1024-question calibration,100-grid search,modality/importance decision and BF16 thresholds,no renormalization | Existing DeepEP-1 sentinel for real skipped execution;vLLM metadata boundary port with mask lifetime corrected to original source | Unreleased public fast CUDA path;single-H200 Figure6 timing;fullbenchmark/Kimi |

No-op controls and exact primitive tests establish decision/math fidelity within
their stated boundaries. They do not certify all autoregressive trajectories
bitwise across runtimes. Native Libra quality checks compare the first greedy
token and final-prefill distributions; route-frozen oracle variants are explicitly
diagnostic rather than model-correctness runs.

## Unexpected slowdowns: do not blame the paper

- SERE's port must collect TP-sharded routes before using the DP-local primary
  union. This cost does not exist in its single-device paper environment.
- MoDES's first port wrongly repeated modality-mask construction per layer. The
  corrected once-forward cache preserves paired natural outputs and recovers
  about9.6% versus that port. This is our engineering correction,not a successor.
- Algorithmically no-op SERE/MoDES settings still have appreciable cost in the
  current Python/vLLM boundaries. Through-first-EOS parity is100% in96 paired
  fixed-output comparisons;after-EOS continuations are not bitwise identical.
- The native Libra log warns that the H100 E=40,N=768 Triton configuration is
  absent and its default is used. Both prediction variants share that native
  expert pool,so the bounded predictor comparison is useful,but absolute paper
  speedup is not validated. No unsafe tuning or assertion bypass was applied.
- All results are scoped to tested Qwen conditions. Kimi is conditional on a
  material nontrivial survivor,not a box to tick before any failure is identified.

Final negative opportunity verdicts,if any,must not be presented as universal
refutations of SERE,Libra or MoDES. Port overhead,original-regime mismatch and
known approximation trade-offs are valid reasons not to promote a successor.
