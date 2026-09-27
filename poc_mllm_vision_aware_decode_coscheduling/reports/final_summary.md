# Final summary

The bounded Stage-A experiment rejects runtime development for this tested method family. Even an oracle that sees every recorded future decode checkpoint reduces the held-out 1-128 expert union by only **2.458% versus FIFO** and **0.263% versus length matching**, far below the required 10%. Moreover, text, full-prefill, and vision-separated policies produce identical held-out waves, so there is no MLLM-specific scheduling increment to test physically.

The conclusion is scoped to this frozen 20-request, wave-4, fixed128 long-reasoning pool, 22 recorded decode checkpoints, and this simple common grouping family. It does not refute the earlier vision-conditioned routing signal; it shows that the signal does not create enough co-scheduling headroom here. The 24-request target was short by four, but the checkpoint-oracle ceiling is sufficiently below the gate that expanding a runtime prototype is not warranted.

BASELINE_STATUS: READY_REUSED_TRUE_EP4

REQUEST_POOL_STATUS: FROZEN_20_DISCOVERY_12_HELDOUT_8; RECOMMENDED_24_SHORTAGE_4

ORACLE_HEADROOM: 2.458%_VS_FIFO; 0.263%_VS_LENGTH; FAIL_10PCT_GATE

TEXT_POLICY_HEADROOM: 2.201%_VS_FIFO; 0.000%_VS_LENGTH

FULL_PREFILL_ELDR_STYLE_HEADROOM: 2.201%_VS_FIFO; 0.000%_VS_LENGTH

VISION_SEPARATED_HEADROOM: 2.201%_VS_FIFO; 0.000%_VS_LENGTH

VISION_INCREMENT_VS_FULL: ZERO; DELTA_EXPERTS=0.000; CI95=[0.000,0.000]; IDENTICAL_HELDOUT_WAVES

VISION_INCREMENT_VS_TEXT: ZERO; DELTA_EXPERTS=0.000; CI95=[0.000,0.000]; IDENTICAL_HELDOUT_WAVES

TRUE_EP4_TPOT_GAIN_VS_FIFO: N/A_STAGE_B_NOT_RUN

TRUE_EP4_TPOT_GAIN_VS_FULL: N/A_STAGE_B_NOT_RUN

FIXED128_E2E_GAIN: N/A_STAGE_B_NOT_RUN

RUNTIME_PARITY: NOT_RUN_STAGE_A_GATE_FAILED

VERDICT: INSUFFICIENT_BATCHING_HEADROOM

NEXT_ACTION: Stop this co-scheduling method family for the tested pool; do not implement a runtime scheduler from this evidence.
