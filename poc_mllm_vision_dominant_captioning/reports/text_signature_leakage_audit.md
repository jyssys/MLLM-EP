
# Text-signature leakage audit

The linguistic prompt bytes are identical, but the text-token router signature is not assumed identical: text-token hidden states occur after image-token context and their positions vary with image grid length.

- P_text within-domain cosine distance: 0.0145
- P_text between-domain distance: 0.0230
- P_text gap: 0.0085
- P_text same-domain NN: 76.000%
- P_vision gap / same-domain NN: 0.1166 / 72.500%

Thus any P_text variation is labeled **image-conditioned contextual/position variation in routed text rows**, not request-specific lexical task information or placeholder misclassification. The hook derives vision rows from the runtime multimodal mask; visual rows are subtracted from all prefill rows before P_text construction.
