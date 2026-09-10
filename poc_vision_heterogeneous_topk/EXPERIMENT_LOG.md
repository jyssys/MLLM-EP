# Experiment log

## Provenance

- Repository checkout: `/home/esjung/MLLM-EP-vision-hetero-topk`
- Starting HEAD: `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`
- Remote `origin/main` observed on 2026-09-11: `e228b44cec1f5ffa32953e52540086423f84f33f`
- Branch: `flashvep/vision-heterogeneous-topk-poc`
- Working spec source: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/bottleneck_reproduction_vision_heterogeneous_topk_poc_spec.md`

All task GPU commands must export exactly `CUDA_VISIBLE_DEVICES=4,5,6,7`.

## Runtime contract

- Model: Qwen3-VL-30B-A3B-Instruct, BF16.
- Model shape: 48 decoder layers, hidden size 2048, 128 routed experts,
  top-k 8, expert intermediate size 768.
- Runtime: vLLM 0.20.0, PyTorch 2.11.0+cu129, CUDA 12.9, NCCL 2.28.9.
- Topology: TP2 / DP2 / EP4; `DeepEPHTAll2AllManager`; Triton
  unquantized fused experts; DBO and EPLB disabled; eager mode.
- Physical GPUs: 4,5,6,7. Topology audit reported NV18 links among the four.
- Task-owned burn was drained before every measurement. No process on GPUs
  0--3 was touched.

## E01 — Clean request bottleneck sweep

- Result: `results/bottleneck_clean_20260911_0041/`
- Workload: output one token, concurrency two, prompt length
  256/512/1024/2048/4096/8192/16384.
- Protocol: one warmup followed by three clean repetitions.
- Median TTFT (ms): 102.759, 98.414, 100.266, 103.370, 104.583,
  185.613, 384.937.
- Result: a long-prefill economic regime exists at 8K and 16K.

## E02 — Observer-heavy MoE decomposition

- Result: `results/bottleneck_stage_20260911_0046/`
- Raw: 98,688 rank-local MoE rows. Same-device CUDA events only.
- The full observer increased TTFT by 120--609%; its wall time is therefore
  excluded from clean request timing. Stage durations are used only for
  critical-rank attribution and then paired with the separate clean TTFT.
- At prompt 8K/16K, summed critical expert time was 73.744/137.660 ms,
  or 39.73%/35.76% of clean TTFT. The load-count-proportional perfect
  balance upper bound was 10.58%/9.45% of clean TTFT.

## E03 — Fresh Qwen3-VL routing/output capture

- Result: `results/vision_capture_fresh_20260911_0050/`
- Nine real images; natural, fine-grained, and chart/document categories;
  336/448/672 edges; 100/196/441 visual tokens.
- Captured layers 4/12/24/36/44/47, actual hidden states, top-8 IDs and
  weights, and expert outputs. The derived route oracle contains 54
  image-layer observations.
- Raw captures remain local and are intentionally not committed; manifest,
  derived tables, and analyses are tracked.

## E04 — Same-budget route-policy oracle

- Script: `analyze_policy_oracle.py`; fresh summary:
  `results/final_analysis/fresh_route_policy_summary.csv`.
- Policies: random, semantic (lowest router weight), and EP-tail-aware
  water filling (lowest-cost removable branch on the current max-loaded
  rank). Text branches and every token's top-1 are retained.
- At 10% vision-assignment removal, median max-rank reduction was 8.12%
  for semantic and 19.48% for tail-aware (2.40x). At 5% it was 4.11%
  versus 12.42% (3.02x). This establishes the intended causal geometry
  effect at exactly matched assignment count, not a usable quality point.

## E05 — Exact captured-layer DeepEP replay

- Script: `exact_deepep_replay.py`.
- Actual Qwen3-VL expert weights, hidden vectors, routes, and DeepEP HT.
- Five warmups; randomized policy order; 30 same-device-event repetitions
  per condition. One native M=221 check and tiled M=4096/8192 stress runs.
- Tested k=8/6/4/2/1, exact removal budgets 1/2/5/10/20/30/40/50%,
  preserved weights and renormalization, layers 4/24/44, and three image
  contents at the final 10/20% gate.
- Renormalization worsened the error frontier and was rejected.
- At 10% budget across camera/cell/chess layer-24 M8192 runs, median
  semantic versus tail-aware MoE reduction was 7.22% versus 11.17%, while
  worst-rank local-output relative L2 was 10.09% versus 13.16%.
- At 20%, the corresponding values were 10.18% versus 13.34% MoE and
  18.96% versus 19.69% relative L2.
- Even 1% removal failed the strict local relative-L2 <=1% screen.

## E06 — Low-overhead attention attribution

- The initial inherited hooks in `bottleneck_low_*` did not attach to the
  live attention path. Those runs are instrumentation failures, not evidence.
- A dedicated nonblocking event hook succeeded in
  `results/bottleneck_attention_20260911_0138/`.
- Observer tax was +7.09% at 256, -0.47% at 8192, and approximately zero
  at 16384. Rank-critical 48-layer attention sums were 31.989, 48.306,
  and 127.778 ms respectively.

## E07 — Decision and implementation gate

- Observer-assisted Amdahl projection used the separately measured clean
  request TTFT; it is a projection, not a measured variable-k E2E result.
- Three-content median projected TTFT gain at M8192:
  semantic/tail-aware = 4.49/6.94% at 10% budget and 6.33/8.30% at 20%.
- The 20% point barely crosses the lower economic screen only by accepting
  about 19.7% worst-rank local MoE relative-L2. Quality-plausible tiny
  budgets remain far below the economic gate.
- Per the working contract, full variable-k vLLM integration and benchmark
  accuracy runs were not started.

## E08 — Full-model logit and short-greedy diagnostic

- Script: `logit_correctness.py`.
- This is a correctness-only Hugging Face forward on physical GPU 4 with all
  four allowed devices visible. It masks retained top-8 weights before expert
  accumulation but deliberately performs all expert work, so it provides no
  performance evidence.
- A three-image screen was followed by nine-image confirmation across all
  capture categories. The policy is applied to vision branches in all 48
  decoder layers; text branches and every visual top-1 remain exact.
- At 10% removal, both P2 and P3 retained the same first greedy token on 9/9
  samples. Four-token greedy exact match was 9/9 for P2 but 6/9 for P3.
  Median logit relative-L2 was 4.23% for P2 and 6.57% for P3.
- At 20%, four-token exact match was 7/9 for P2 and 6/9 for P3. Median logit
  relative-L2 was 5.39% and 4.38%; the P3 median KL was higher (0.00281 versus
  0.00143). Non-monotonic aggregate norms do not override the sequence-level
  failure distribution.
- Result: `results/logit_correctness_nine_20260911_0202.summary.json`.
