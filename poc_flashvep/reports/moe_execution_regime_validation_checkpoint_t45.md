# T+45 checkpoint — MoE execution-regime validation

Timestamp: 2026-09-05 14:10 KST (approximately 48 minutes after sprint start)

## Strongest signal

With global/per-shape warmup, deterministic shuffled interleaving, and 20–30 paired
measurements, the old M=128 versus M=512 sign flip does **not** survive.  At M=128,
F4/F1 expert latency is approximately −1.6% (30-repetition interleaved run), while
at M=512 it is +14.2%.  The M=512 effect is repeated at layers 4/24/44 (+13.7%,
+14.2%, +15.9%) and in the active-expert disentanglement run (+12.7%, +16.2%,
+26.6% for A8/A16/A32).  Dispatch moves in the same direction; combine often
counteracts it, so full critical wall time is only about +1–5%.

## Strongest negative

At fixed M=512, F2 geometry with balanced aggregate rank load (pair-concentrated
versus cyclic destination pairs) differs by only +0.35% expert and +0.90% wall
(20 paired repetitions).  H7 distribution-only control is also weak (≤3.7% expert,
≤1.8% wall after interleaving).  These controls argue against a simple traffic-shape
or expert-count explanation.

## Unresolved confound

The block-ordered 10/30-repetition runs still contain large state/order outliers
(notably A16 at M=512/1024).  Interleaving removes the original sign reversal but
does not fully identify the source of the residual M=512 fanout penalty.  Local-only
expert diagnostics (+22.7% at M=512) and DeepEP dispatch (+17.9%) suggest an
expert-kernel/packing interaction; combine offsets it.  The local diagnostic adds a
second timing path and is therefore treated as supporting, not primary, evidence.

## Completed live experiments

H1/H2 randomized grids, H1 interleaved sign-flip pairs, H2 active×fanout
disentanglement, H3 local-vs-DeepEP diagnostic, H4 alignment focus, H5 layers 4/24/44,
H6 balanced destination geometry, H7 distribution shape, H8 real-route transfer, and
H10 generic Qwen3-30B controlled check.  All completed runs reported correctness=true
and used physical GPUs 1–4 only.

## Remaining experiments / analysis

1. Consolidate H1–H10 CSVs and generate a single causal scoreboard.
2. Add a second M=512 interleaved replication if time permits to quantify variance.
3. Add a concise kernel/phase interpretation and real-route limitation (natural
   routes are already F≈3.4–3.7; no matched F1/F4 pair).
4. Produce final report, gate summary, figures, commit, and push after the minimum
   75-minute exploration window.

Status: CONTINUE LIVE VALIDATION; do not treat the old sign flip as reproduced.
