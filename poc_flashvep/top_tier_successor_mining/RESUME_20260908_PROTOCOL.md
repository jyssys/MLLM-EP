# Deferred GPU validation resume — 2026-09-08

Resource scope: physical4,5,6,7 only, TP2/DP2/EP4 for vLLM; native Libra keeps
its author four-rank runtime and is labeled separately. No work on0–3.
User-requested idle/final utilize runs use `VLLM_UTILIZE_GPUS=4,5,6,7` with the
audited shell script. Burn is stopped before experiments and never evidence.

## Execution order and equal-screening gate

1. Finish interrupted official MoDES1024/grid100 search; repeat one cached point
   before accepting old evaluations on the new devices. No threshold chosen using
   held-out labels. Keep the partial checkpoint and interruption record.
2. Exact32-request full-VL capture, native48-layer text and VL parity. A failed
   VL bridge is PORT_FAILURE, not Libra failure. Clean native timing follows only
   numerical checks; post-vision prefill is not full-request E2E.
3. Official-norm SERE and completed MoDES thresholds on matched held-out GQA and
   ChartQA; natural greedy generation and unchanged official task scoring.
4. Fresh sentinel/duplicate-route parity, actual vLLM DeepEP activation proof,
   then clean direct-request paired timing. Global warmup plus per-policy warmup,
   randomized policy order, repeated matched request cohorts. B1/B4/B16 are
   DP-local concurrency, with two DP sources recorded explicitly.
5. All-three comparison, trivial-fix and headroom gates. Only a real surviving
   failure justifies conditional Kimi and a successor mechanism.

## New measurement control, not a successor method

Local vLLM0.20 `EncoderCacheManager` retains image embeddings across requests;
`inputs/llm.py::MultiModalUUIDDict` defines user UUIDs for all multimodal caches.
Primary resumed online runs therefore use unique per-request image UUIDs and
identical pixels. This removes A/B image-cache-order confounding. Optional shared
cache runs remain separately labeled. Prefix caching remains off in either case.
Do not combine previous shared-cache timings with the new cold-image primary.

Request latency begins immediately before engine.add_request and includes its
input processing, queueing, encoding, prefill and natural decode. Image file I/O
and chat-template preparation occur before submission for all policies. Hardware
model-load/compile warmup is outside measured request latency.

### Live-renderer correction

The first resumed raw-string-prompt path silently dropped UUIDs in local
`inputs/preprocess.py::_process_text` (the token-prompt branch does not).
Observed MM cache hits>80% falsified the intended cold contract. That incomplete
run is explicitly INVALIDATED and excluded. Primary runs now use the current
`LLM._preprocess_cmpl_one` renderer path, INSIDE the request timer, record actual
`mm_hashes`, and assert no cold hash is reused. Fresh60-request sanity has60
unique hashes and0.0% reported MM cache hits. This is measurement repair, not a
successor optimization and not a baseline-speed result.

Core token-ready timestamps are primary; frontend receipt is separately retained.
Source: vLLM/v1/engine/EngineCoreOutputs.__post_init__ uses host time.monotonic;
v1/metrics/stats.py updates RequestStateStats.last_token_ts from that timestamp;
output_processor attaches those stats to RequestOutput. Because a synchronous
frontend submits the whole cohort before polling, its receive time can be late
without the request still executing. We record BOTH and do not call core-ready
latency HTTP/client-delivery latency. No cross-GPU CUDA timestamp subtraction.
The first live sanity must verify nonzero monotonic per-request timestamps.

Known limitations remain explicit: the author Libra harness supports equal source
lengths and prefill; the VL bridge supplies exact captured encoder outputs.
HF replica wall time is quality diagnostics, never claimed as EP speedup.
