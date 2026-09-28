
# Q2 visual dependency signal

VDS-A and VDS-B were measured inside the actual decode FlashAttention path on TRUE EP4 for frozen layers [4,12,24,36,44,47] and steps [8,16,32,48,64,96,112]. The hook reconstructs attention to the current paged KV cache and never changes model outputs. It is diagnostic-only.

- VDS points: 1,122; joined functional paths: 256/256.
- Frozen primary before damage analysis: `vds_attn_mean`.
- Held-out VDS-A/damage Spearman: -0.2548.
- Held-out VDS-B contribution/damage Spearman: -0.0644.
- External caption image-swap causal lane: expert change 5.359%, EP4 changed-rank 1.631%.

The causal lane proves image-conditioned routing exists, but it is not observation-aligned with this 32-request damage cohort. Direct VDS-A/B↔VDS-C alignment is therefore **not proven**.
