# Bounded setup checkpoint — 2026-09-08 19:38 KST

The pinned FlashAttention build has run for three hours (started ~16:38 KST).
It has completed 278/399 compile steps without a compiler failure. This is
CPU compilation, not GPU research time and not evidence against Layered Prefill.

Do not spend the rest of the session actively debugging or rebuilding this
dependency. The already-progressing compile may finish in the background while
the other two native baselines and clean vLLM transfer traces collect useful data.
An unchanged build is less risky than deleting its ephemeral editable-build
directory or replacing its numerical kernels. No original environment is changed.

Alternate faithful diagnostic in parallel: measured Qwen3-VL TP2/DP2/EP4 layer,
attention, router, dispatch, expert and combine costs with request joins. A
contiguous measured-cost partition oracle can screen whether an MLLM port is
worthwhile. This is labelled a transfer oracle, never native Layered Prefill
request performance. Native official TP2 Qwen3 smoke remains mandatory if the
build completes. Environment/port limits and method failure stay separate.
