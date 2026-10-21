# Prospective numerical protocol

Historical strict C2 retains its original limits and P512 FAIL.
No new tensor-error PASS threshold is defined from test answers.
First compare HF, custom unsharded, custom TP4, native TP1, and native TP4 on synthetic locked prompts.
Use P128/P256/P512/P2048 and inspect all loops/layers at sampled positions.
At P512 retain zero-based steps 18, 19, 26, and 27.
Separate backend differences from same-engine TP1/TP4 differences.
Until independently calibrated FP32 checks justify qualification, native/custom TP quality uses EXPLORATORY status.
Exploratory generation requires finite logits, valid vocabulary IDs, audited weights, and live greedy feedback.
Quality and numerical status remain independent. Task accuracy never retroactively passes the old gate.
No exploratory timing constitutes qualified architecture speedup.
