# Correctness

The existing-backend comparison uses identical synthetic BF16 inputs, weights,
expert offsets, SwiGLU sequence, route weights and packed-row order. Across all
three dense and all three compacted restarts, PyTorch grouped versus vLLM fused
has:

- maximum absolute difference: 0;
- maximum relative L2: 0;
- cosine: 1.0 in the recorded cases.

The route-corpus invariants also preserve every token-expert assignment and the
top-k=8 count. [KERNEL_CORRECTNESS.csv](../KERNEL_CORRECTNESS.csv) contains the
per-case checks.

No custom kernel entered the model, so there is no RefineGEMM trajectory whose
quality can be claimed. The unchanged strongest baseline has deterministic
three-restart output hashes, GSM8K 5/32 with NFE66 and HumanEval 6/32 with
NFE86. These bounded scores are substrate anchors, not quality evidence for a
new method. The fresh physical-GPU4--7 GSM8K run also has NFE66 and all 32
answer fields exactly match the prior anchor.
