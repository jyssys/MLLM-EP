# Progress log

## Completed

- Read the complete 348-line English specification and the reporting-skill references.
- Created branch `flashvep/ouro-chunk-loop-slo-sensitivity-poc` without changing the 68 pre-existing staged entries.
- Checked eleven historical files against the frozen commit and verified 375 raw archives.
- Recomputed 11,172 historical absolute-SLO cells from actual request timestamps.
- Located all eight historical mismatches. Every affected request was background decode, not the new long prefill.
- Completed corrected GPU4-alone and four-worker calibration. Preserved v1 lifecycle failures.
- Froze 5×/10×/25× TBT tiers at 135.946614/271.893227/679.733069 ms.
- Completed 204 new counterbalanced fixed-grid cohorts across twelve process blocks.
- Froze 148 public conversational/article prompts before natural-response comparisons.
- Completed seven natural-stop pilot policy cells. All eight requests per cell reached observed chat-turn stops.
- Preserved the no-chunk cleanup failure before its following staged-G4 case.

## Registered open-loop execution

Held-out open-loop Poisson trials use four independent GPU workers.
The four frozen offered rates are 0.125/0.25/0.5/1.0 requests per second.
Every GPU/process block runs every shortlisted policy in randomized order.
Common request families, arrival offsets, and seeds support paired comparisons within each block.
All sixteen Poisson job blocks and six burst job blocks completed.
The first wrapper transition failed after Poisson capture and restored the recognized burners.
The resumed controller skipped completed evidence and executed the original remaining manifests.
The preserved transition transcript and timeout records remain available.

## Final completion

1. The empirical Atomic-only reference is frozen at 0.25 offered requests per second under the registered conservative finite-window rule.
2. Ten paired shortlist cohorts across six process blocks are complete. Full-grid cells retain three cohorts.
3. Expanded numerical comparisons passed 48/48 exactly for tokens, depth, hidden states, logits, and KV.
4. Completed 128 Poisson and 48 burst policy runs. All 6,880 offered instances remain in the denominator.
5. Retained 6,171 complete natural responses, 707 caps, and two timeouts. Admission errors: zero.
6. Completed four Nsight captures, separate event profiles, twelve paired observer cohorts, and the quiet GPU4 policy control.
7. Validated all 71 expected jobs and full-R4 work for 6,878 completed serving instances.
8. Reviewed all-offered confidence intervals, figures, independent decisions, and correctness limits.
9. Verified 1,834 lossless archives. Local raw originals remain intact.
10. Restored recognized burners on GPUs 4–7. The final delivery step commits and pushes only this study.

Do not resume native-vLLM Loop PP.
Do not build Adaptive Joint or claim multi-GPU serving capacity from independent experiment workers.
