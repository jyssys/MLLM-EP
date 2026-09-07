# Rethink after 40 assumptions

**Expected:** model/modality contracts might yield a new MLLM-specific axis.

**Observed:** image effects collapse to 0.3--2% after controls; speculative
and coalescing paths fail quality/headroom.
**Failed assumption:** relaxing model semantics creates a useful distributed
execution window for free.
**New system fact:** Qwen-VL uses a shared exact MoE contract; modality is not
causal in the tested regime.

New children: (1) request-class execution contracts, (2) attention/MoE joint
resource state, (3) spatial representation ownership.  The first and third
are crowded/low-headroom; the second is a generic observability question with
no measured intervention.
