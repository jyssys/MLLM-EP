# Libra reproduction status at GPU release

## Final GPU resumption supersedes the historical checkpoint below

Final resumption: native48-layer execution and eight real VL groups completed. Internal first-token agreement320/320;HF reference31/32 distinct inputs,so the bridge remains APPROXIMATE. Valid frozen-current-route prediction gain0.146%;actual overlap measured with Nsight. Large no-hook native control passes80/80;large oracle-interleaved run is invalidated. No full-request VL E2E or Kimi claim. See NATIVE_OVERLAP_20260908.md.

## Historical audit / release record

ACTUAL_SUPPLEMENT_FUNCTIONAL_PASS; FULL_MLLM_AND_PERFORMANCE_DEFERRED.

User-supplied source is available and supersedes the earlier public-repo404
fallback. Original Cython planner compiled unchanged and48 invariants passed.
Native SGLang0.4.10 executes its actual AllGather/local/remote/AllReduce and
prefetch-buffer path on physical1–4, using a separate Python3.10 environment.

Real Qwen text weights, four layers:24/24 measured rank comparisons have equal
first greedy tokens. Minimum logits cosine.9997333, maximum abs.40625 and maximum
relative L2.0231472. This is NOT bit-exact and NOT a full-model quality test.
Concurrent MoDES calibration excludes all native timings from speed claims.

The VL bridge has six CPU boundary checks, covering MRoPE arithmetic and DeepStack
residual order. Its32-request exact-length manifest exists. The GPU tensor capture
and48-layer native runs were queued but cancelled before launch at user release.
Do not report that the full VL port ran.

Original paper model/hardware scale and author prefill harness differ materially
from the small Qwen transfer; see PAPER_AUDIT and VL_BRIDGE_CONTRACT.
