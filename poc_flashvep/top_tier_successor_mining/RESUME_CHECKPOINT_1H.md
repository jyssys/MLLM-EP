# GPU resumption checkpoint — 2026-09-08 ~12:08 KST

Physical4–7 only. No final ranking yet; EP request timings and native VL parity
are still pending. Idle burn has not run during research measurements.

## Completed fresh evidence

- MoDES official1024/grid100 frontier:436unique pairs, completed52.34min resumed
  replica interval. Exact cached-point parity before reuse. Target70/85 realized
  skip71.328/85.267%, calibration KL0.011076/0.017023.
- Exact full-VL32-request input/route/logit capture completed.
- Native supplied Libra text48layers/M128:60measured rank rows across15paired
  execution units; all greedy-first matches, min cosine0.999756. Median prefill
  vanilla80.779ms, Libra173.627ms. Small shape is outside the paper's stronger
  large-model/long-prefill regime; this is NOT an MLLM-specific failure finding.
- Official-norm SERE + completed MoDES held-out B1 quality:128ChartQA and128GQA;
  independent-image clustered CIs, full-prediction official scoring.

| Policy | ChartQA delta pp (95% CI) | GQA delta pp (95% CI) |
|---|---|---|
|SERE S2/rho.5|-11.71875 [-18.75,-5.46875]|-8.59375 [-14.0764,-3.4953]|
|SERE S4/rho.5|-0.78125 [-2.34375,0]|0 [-3.0303,3.1502]|
|SERE S2/rho.7|-0.78125 [-4.6875,3.125]|+0.78125 [-3.1752,4.6154]|
|MoDES target70|-2.34375 [-7.03125,2.34375]|-0.78125 [-5.46875,3.9683]|
|MoDES target85|-4.6875 [-8.59375,-0.78125]|-0.78125 [-5.46875,3.8469]|

B1 vanilla89.0625% ChartQA,56.25% GQA. SERE B16 faithful cohort control:
S2/rho.5 delta0pp on BOTH tasks (Chart CI[-2.34375,2.34375], GQA
[-1.7699,2.3256]). B16 vanilla89.84375/57.8125%. Do not conflate batch-induced
quality recovery with retained EP speedup; that is the next measurement.

## Port issue isolated, not method failure

VL native load failed because this checkpoint stores packed expert matrices in
right-multiply layout: gate_up[E,H,2I],down[E,I,H]. The bounded bridge incorrectly
assumed standard Linear layout. A CPU test using a real expert reproduces the
shape failure; split-last-dimension plus transpose fixes it, reproducing expert
math within6.56e-7maxabs. Author planner/kernels/streams unchanged. Full GPU
parity is being rerun; not yet accepted based on this component test alone.

## Next

1. Full VL parity, then long-text and natural-VL native prediction-cost controls.
2. Fresh DeepEP compatibility/hook/timestamp sanity.
3. Clean natural-EOS EP B1/B4/B16 quality+request timing; fixed-length control.
4. Only then comparable all-three scoring and conditional deep dive.

ACE(arXiv2609.05228v1) was newly audited as an adjacent calibration-free scoring
collision. No successor has been implemented or promoted.
