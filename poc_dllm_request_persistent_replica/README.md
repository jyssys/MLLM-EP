# Request-Persistent / Cross-Block Expert Residency

Stage A characterization grounded in LLaDA2.0-mini physical EP4. The main
comparison is cold-once request-specific residency versus **prewarmed Global**,
not versus Vanilla. Read [the report](reports/report.md),
[frozen cohort](results/frozen_manifest.json), and
[authoritative specification](references/authoritative_spec.md).

Authoritative audited summary: `results/summary_v3.json`; corrected CI:
`results/bootstrap_ci_v3.json`. Numerical byte advantage exists but is **not
actual-load matched**. Apparent oracle recovery must not be read as a passing
gate. Historical caps fail on future iterations for Global and causal plans.
Preliminary versions are preserved; no frozen policy is retuned.
Audited figures are in `figures_v2/`.

Only physical GPUs `2,3,6,7` are authorized. Actual topology is dense TP4 /
source-SP4 / physical EP4 / DP1 / PP1. This is not Qwen's TP2-DP2 topology.
The existing validated `poc_dllm_ep_comm` worker and
`poc_dllm_block_local_replica` accounting/planning/P2P evidence are dependencies
available in the parent commit. External dInfer code is not modified.

## Reproduce offline

From the repository root:

```bash
bash poc_dllm_request_persistent_replica/scripts/reproduce.sh offline
```

This creates a fresh temporary artifact directory, restores the complete
lossless raw evidence, revalidates destination-deduplicated physical accounting,
rebuilds planner-only archives, reruns short/long policies, and generates CI,
figures and reports. It does not use GPUs. CPU Python default is
`/home/esjung/anaconda3/bin/python`; override with `DLLM_PERSIST_CPU_PYTHON`.
It refuses to overwrite existing result paths. Simulation may take tens of
minutes on a small CPU allocation. `--workers 4` is CPU process parallelism,
not concurrent GPU inference.

Complete committed raw is in `raw/compact`: original JSON bytes and every named
NPZ array are recoverable. Empty/repeated arrays are losslessly content-deduped.
Original ZIP file byte identity is not required. Local original capture files
remain preserved under ignored `raw/long_v1`; derived ID-only duplicates are
also left on disk and rebuildable. `raw/compact/index.json` records hashes,
array counts and verification status.

## Recollect physical evidence

```bash
bash poc_dllm_request_persistent_replica/scripts/reproduce.sh capture
```

Uses frozen requests, unchanged native decoding, the existing GPU Python
`/home/esjung/.venvs/llada2-flash-sglang-053/bin/python`, the pinned model and
existing dInfer checkout. Watchdog/finally cleanup stops only authorized burner
processes, runs current EP4, then restores authorized burners. The shared
`run_utilize.sh` default may target a different GPU set: it is always invoked
with **`VLLM_UTILIZE_GPUS=2,3,6,7`**, never without the override.

Budget512 can finish the native diffusion block and return up to543tokens;
EOS behavior is not altered or cherry-picked. The 32request cohort is8cal/24held;
primary coding, reasoning secondary. Three warmups, three lanes/request and two
extra acceptance probes total101model runs.

## Interpretation and implementation boundary

- `B_live_only` removes context and already-accepted traffic structurally;
  this is not an Epoch execution/speedup.
- Replicas are layer/source/expert-specific; exact loaded expert size6MiB.
- Source→destination hidden rows dedup within destination; partial coverage of
  an expert bundle does **not** close its remote communication edge.
- All original K8 branches execute fresh. No output reuse or routing changes.
- Global copies are prewarmed/free; causal profiles full one/two blocks and
  copies once/request. No per-block reload or learned predictor.
- Historical load caps are not future guarantees. Actual worst layer/iteration
  load is separately reported; no held-out truth prunes a causal plan.
- Calibration freezes candidate/control before held-out summary. Bootstrap unit
  is request, not block/layer/iteration.
- Copy milliseconds use measured P2P. Byte ROI is **not** time ROI. Avoided
  communication time and replica speedup remain unresolved unless Stage B runs.
- Stage B must not be built if any central Stage A premise fails.

Tests:

```bash
/home/esjung/anaconda3/bin/python -m pytest -q poc_dllm_request_persistent_replica/tests
```

Tests cover source/destination closure, layer identity, copy-once lifetime,
future-data invariance of causal selection, observed-history load caps, exact
cached/reference planner parity, finite JSD and lossless raw round-trip.
