# dLLM intra-refinement Attention–MoE pipeline feasibility

Read [report.md](report.md), [dependency_audit.md](dependency_audit.md), and
[prior_overlap.md](prior_overlap.md). Authoritative specification is copied
byte-for-byte in `references/authoritative_spec.md`.

This branch reuses the validated dInfer/DeepEP worker setup from
`poc_dllm_ep_comm` and adds only PoC-owned observers and component probes.
No external model/runtime source, weights, routing or acceptance rule is edited.

Analysis reproduction:

```bash
bash poc_dllm_attn_moe_pipeline/scripts/reproduce.sh
```

Fresh acquisition (requires assigned GPUs **2,3,6,7 only**, the pinned local
model snapshot and dInfer environment; refuses competing processes):

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python \
  -m poc_dllm_attn_moe_pipeline.scripts.collect_readiness \
  --output reproduce_new --limit 12
```

Use a fresh output directory. Metadata selection is already immutable in
`results/main/design_freeze.json`; do not rerun `inspect_runtime` over it.

`main_v1` retains native12-request early-kill traces and an aborted unsupported
graph-capture probe. `components_v2` retains a failed optional-mask diagnostic.
`components_v3` supplies the completed calibration-only producer split and
partial-attention references. `route_sanity_v4` supplies fresh actual source
expert IDs/weights and owner-local receive-ID validation. Hidden snapshots are
RAM-only, not committed. No candidate overlap runtime exists. All captured raw
JSON and failed logs are committed as byte-exact gzip under `evidence/`, with
SHA256 round-trip checks. Reproduction restores local ignored `raw/` without
overwriting divergent files; originals remain preserved on the server.
`results/main/integrity_audit.json` validates 430 lossless archives, actual
layer-specific EP ownership, route ordering and 96 native generate calls
(warmup/smoke/retries included). The 64-token requested native budget is rounded
to block32; the twelve task suffixes actually contain 64–93 returned tokens.
These are bounded prefixes, not task accuracy scores.

Unknown clocks/latent completion costs are null, not zeros. The zero native
window means whole-buffer publication, not zero hardware row skew. Engineered
producer windows do not establish a measured full two-layer seam.

Per authoritative spec§6, restore only pre-existing authorized burner state;
never run an unmasked default burner on other GPUs. Initial state here wasOFF.
