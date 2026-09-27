# Mixed-domain online calibration PoC — final summary

The bounded study completed a frozen 160-request, four-domain TRUE EP4 trace, a balanced 48-request TRUE EP2 anchor, no-future-leakage online replay, and mandatory EP8 structural replay. Including bounded smoke, it used 210/400 full-model request executions. It implemented no routing change, expert skipping, or production scheduler.

Primary result: **GENERIC_FULL_PREFILL_SUFFICIENT**. At the discovery-selected 10% balanced warm-up, held-out `P_separated` expert-histogram cosine is 0.9560. Its beneficial delta over `P_full` is -0.0003 (95% CI [-0.0013, 0.0007]); over `P_text`, -0.0014 ([-0.0027, -0.0002]). Thus modality separation adds no held-out value, while the simpler text-only signature is statistically better on this metric.

The interpretation follows the pre-frozen criteria. Expert-level gain, rank-level gain, domain-label control, and locality are not conflated. EP8 is never described as measured latency.

```
BASELINE_STATUS: READY_REUSED_AND_TRUE_EP2_EP4_RECONFIRMED
DOMAIN_COHORT_STATUS: FROZEN_160_REQUESTS_40_PER_DOMAIN_IMAGE_HASH_DISJOINT
DOMAIN_SIGNATURE_SEPARATION: P_SEPARATED_HELDOUT_GAP=0.0929; NN_SAME_DOMAIN=73.75%
WARMUP_RATIOS_REALIZED_PER_DOMAIN: 0%=0/40; 1%=1/40(2.5%); 5%=2/40; 10%=4/40; 20%=8/40
BEST_WARMUP_RATIO: 10%_DISCOVERY_FROZEN_KNEE
FROZEN_WARMUP_GAIN: EXPERT_COSINE_DELTA_VS_UNCALIBRATED=0.5254
CONTINUAL_CALIBRATION_GAIN: IID_EXPERT_COSINE_M2_MINUS_M1=0.0038
P_TEXT_RESULT: EXPERT_COSINE=0.9574; RANK_COSINE=0.9973
P_FULL_ELDR_STYLE_RESULT: EXPERT_COSINE=0.9563; RANK_COSINE=0.9972
P_SEPARATED_RESULT: EXPERT_COSINE=0.9560; RANK_COSINE=0.9972
VISION_INCREMENT_VS_FULL: EXPERT_COSINE=-0.0003_CI95=[-0.0013,0.0007]
VISION_INCREMENT_VS_TEXT: EXPERT_COSINE=-0.0014_CI95=[-0.0027,-0.0002]
DOMAIN_LABEL_CONTROL_RESULT: P_SEPARATED_BENEFICIAL_EXPERT_COSINE=0.0034_CI95=[0.0010,0.0060]
MIXED_DOMAIN_ORACLE_LOCALITY_HEADROOM: B8_EP4=6.52%
P_FULL_LOCALITY_HEADROOM: B8_EP4=3.79%
P_SEPARATED_LOCALITY_HEADROOM: B8_EP4=3.73%
EP2_PHYSICAL_RESULT: TRUE_DEEPEP_EP2_48_REQUESTS_64_EXPERTS_PER_RANK
EP4_PHYSICAL_RESULT: TRUE_DEEPEP_EP4_160_REQUESTS_32_EXPERTS_PER_RANK
EP8_STRUCTURAL_RESULT: STRUCTURAL_EP8_PROJECTION_ONLY_16_EXPERTS_PER_RANK_NO_LATENCY
EP_SCALING_INTERPRETATION: CHANGED_RANK_EP2=2.11%; EP4=3.32%; EP8_STRUCTURAL=4.12%
VERDICT: GENERIC_FULL_PREFILL_SUFFICIENT
NEXT_ACTION: Do not promote a vision-separated method; if prediction work continues, use the simpler P_text/P_full control in a separately gated PoC.
```
