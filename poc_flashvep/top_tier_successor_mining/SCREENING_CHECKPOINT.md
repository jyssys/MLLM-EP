# Equal-milestone screening checkpoint — 2026-09-07 15:29 KST

No winner selected. No successor mechanism implemented. These are baseline probes,
not a final reproduction or a new-paper claim.

| Method | Paper/source | Mathematical sanity | Fresh MLLM screen | Remaining critical evidence |
|---|---|---|---|---|
| SERE | Official source available, CUDA rerouter built unchanged | 128 route parity tests pass; 400x128 FineWeb-Edu calibration completed | ChartQA batch-1 loses 19.53pp, batch-16 loses 1.56pp on 128 paired items | Official-BF16-calibration replication; real serving quality/speed; token/control-flow/content controls |
| Libra | Final paper/algorithms read; advertised official repo 404 | Actual next-layer gate and planning invariants pass | Across all 47 predicted layers, text top-k recall .855 and vision .791; modest count-locality gap to perfect prediction | Actual copy/local/remote timing and quality-exact runtime behavior; cannot treat missing code as method failure |
| MoDES | Official reference mathematical path available; fast code not released | Four threshold math/no-op tests pass; official ablation behavior checked | Coarse pilot at ~90% calibration skipping loses 11.72pp ChartQA | Full 1024-example official calibration and 100-grid frontier, held-out benchmark, efficient EP sentinel correctness |

## SERE result decomposition

Paired 128 ChartQA items, official relaxed scoring, short-answer prompt, greedy.
The early phase-control result localizes the loss to **decode**, not vision-token
prefill. S2/rho=.5 prefill-only is -0.78pp; decode-only is -19.53pp. This rejects
the attractive but unsupported hypothesis that merely disabling visual prefill
rerouting would solve the observed issue.

Batch 16 largely removes this loss. This is consistent with SERE's explicitly
batch-global primary union and does not yet establish a novel limitation. The
paper's quality batch is 16; our batch-16 result is the more relevant baseline
sanity. Any successor must beat batch/threshold/recalibration controls at matched
quality and serving speed.

Termination diagnostic (not a replacement evaluator): mean generated length rises
from 4.63 to 6.62 tokens with all-phase SERE and 7.24 with decode-only SERE. Of 26
baseline-correct/decode-SERE-wrong answers, first-line-only scoring would recover
10, but only 6 lack EOS by the 32-token cap. Therefore it is incorrect to attribute
the entire quality loss to EOS failure. Numeric/content errors remain. Direct
request latency still needs fresh EP measurement; token count is not E2E gain.

The streamed FP32-Gram calibration differs from official BF16-distance arithmetic:
mean similarity error .000955, maximum .006770; .258% of pair entries cross rho=.5.
A new 400x128 BF16-subtraction/BF16-norm calibration completed in 309.9 seconds.
Use this stronger-fidelity table for confirmatory experiments.

## MoDES reproduction controls

The official 1024-image-disjoint-GQA calibration completed on three independent
quality replicas, with raw-question prompt and official answer-logit mask.
The earlier 64-example short-answer calibration is only a pilot. Official skipped
MLP ablation copies the normalized input instead of zeroing it; a paper-zero
control changed layer importance substantially. Correcting that alone would be
an engineering fix, not a successor research contribution.

## Runtime status

Fresh EP4 runtime proof observed DeepEPHTPrepareAndFinalize + TritonExperts on all
four ranks, DP2/TP2/EP4, BF16, DBO off. A first hook config-context error was fixed.
The direct-output harness also needs internal versus external request-ID joins;
this is instrumentation work and cannot be reported as a paper baseline failure.

Next: finish efficient baseline parity, full MoDES frontier, Libra's real transfer
cost, then all-three failure/headroom/prior-art matrix before ranking.
