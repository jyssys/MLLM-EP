# Causal predictor

Status: **NOT_RUN**.

The spec permits a small supervised predictor only after a meaningful depth-specific oracle result.
This study does not establish that prerequisite.
It therefore fits no prompt-only depth predictor and trains no RL policy.

A future admission predictor could use prompt tokens and request metadata available before prefill.
It could not use future output length, true future exit depths, or post-prefill hidden states.
Past decode observations would not permit free cross-GPU request movement.
The present pilot traces remain evaluation evidence, not arrival-time features.
