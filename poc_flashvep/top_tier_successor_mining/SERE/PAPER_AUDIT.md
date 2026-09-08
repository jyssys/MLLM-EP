# SERE — paper audit

Final reproduction/status update: see REPRODUCTION.md and the main report;
the pending wording below is the preserved pre-screen paper audit.

Source: [ICLR 2026 paper](https://arxiv.org/abs/2602.07616), local PDF/text and
hashes under the result root. Main method, evaluation, calibration ablations and
prefill appendix inspected. Status: PAPER UNDERSTOOD; reproduction pending.

## Execution contract

For each layer, retain the union of each token's top-S experts in the **current
batch**. Each secondary expert maps to the most similar retained expert, unless
the calibrated similarity is below rho. Original routing weights remain unchanged.
Similarity is computed from all experts' responses to shared calibration inputs;
the recommended Frobenius version normalizes distances by the largest pair distance.
The paper uses FineWeb-Edu 400 sequences x 128 tokens; public script defaults to
200 x 64 and processes only one batch. Record this difference rather than silently
treating defaults as the headline calibration.

## Scope that must not be misrepresented

- The reported system uses single-H20 vLLM serving, not distributed EP4.
- Speed metric is TPOT; inputs/outputs fixed 128/32, 5,000 requests for speed.
- Quality evaluation uses longer generations, batch 16, stochastic settings.
- Appendix C.2 explicitly says FLOPs do not decrease and prefill speedup is not
  expected. Primary experts largely saturate during prefill.
- Qwen top-2/rho=.5 is already below vanilla on some tasks; a successor must show
  an additional material limitation, not rediscover that approximation can hurt.
- Calibration data/size/metric robustness and decode-only versus all-phase
  application were already ablated. Simple vision recalibration is an attack
  baseline, not presumed novelty.

## Hypotheses to measure, not conclusions

1. Whether a batch-global primary set remains a useful memory-working-set reduction
   with MLLM token/request composition and distributed source-local batches.
2. Whether output-distance ordering on generic text transfers to visual-conditioned
   **generated text**, separately from actual prefill vision tokens.
3. Whether the same target's quality changes materially with unrelated co-batch
   inputs; hold target, total M and generation settings fixed.
4. Whether any error is fixed by rho/S or calibration changes at matched speed.

No observed failure or successor headroom claimed yet.
