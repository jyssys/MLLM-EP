# Reproduction

## Environment

Use `/home/esjung/.venvs/looped-pd-abc/bin/python`.
Run commands from `/home/esjung/MLLM-EP-github`.
Keep the private RLT checkout at `/home/esjung/external/looped-pd-abc-rlt`.
Do not replace its source with a fresh upstream checkout.
The source contract records the required local patches.

The study base is `b53a8fed36a1706ac74754b53753f20edb8b913e`.
The first implementation checkpoint is `28b1cbf75b0b26a758e98e26c60af9860e7c498e`.
The final summary identifies any later tested changes.

The main cut map uses a matched completed-prompt snapshot control.
The native PD_G1 streaming control remains separate.
Do not combine their measurements under one baseline label.

The original `preflight` command requires the initial base HEAD.
It must not silently bless a different starting checkout.
Its saved `configs/source_contract.json` records the completed audit.
Later revisions must recheck the saved source hashes and the unrelated Git index.

## CPU tests

```bash
PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github \
  /home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.test_frontiers

/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.index_guard
```

The frontier tests require no CUDA execution.
They do not replace model-level KV and output checks.

## New GPU run namespaces

Never rerun into a completed or failed raw namespace.
Create a new configuration name with `configure`.
The dispatcher skips completed runs and refuses to overwrite other existing namespaces.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.configure \
  gate_ouro_L1536_cut2.json reproduction_ouro_L1536_cut2

/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.dispatch \
  reproduction_ouro_L1536_cut2.json --parallel 1 --quiet
```

The configuration preserves GPU IDs, seeds, chunk sizes, and output caps.
Use `configure` overrides only for a separately documented experiment.
The dispatcher validates burner ownership before it frees a GPU.
Quiet runs stop only validated, owned burner sessions across the shared fabric.
The dispatcher restores those sessions after its jobs finish.
It does not restore burners into GPUs with surviving experiment workers.

The complete command ledger is `commands.jsonl`.
Each run retains `arc_contract.json`, worker mapping, source archives, output IDs, and stage counters.
The legacy `configuration.json` also records launcher arguments.
Its `rate=0` argument does not describe the custom arrival schedule.
Use `arc_contract.json` and request timestamps for the actual arrival contract.
The log directory preserves every failed launch.

The decisive fixed-cut configurations are `map_*_decisive_start*.json`.
The additional frozen-choice hold-out configurations use `frozen_choices_start3`.
The native streaming controls use `native_G1_start3`.
The long subset uses `L1536_subset_start0`.
Each name identifies a separate raw directory.

## Analyze actual evidence

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.gate \
  gate_ouro_L1536_cut4 gate_ouro_L1536_cut2 \
  --output gate_ouro_L1536.json

/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.campaign_gate ouro

/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.campaign_gate huginn

/home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.adaptive_recurrence_cut.analyze
```

The analysis uses only completed map namespaces.
It preserves request latency distributions and observed token-delivery intervals separately.
Multi-token output notifications do not supply individual token timestamps.
The summary reports their count instead of fabricating missing intervals.

The following commands produce the remaining measured-data tables.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.migration_analysis
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.memory
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.native_comparison
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.regime_selection
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.uncertainty
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.profile_analysis
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.profile_timeline
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.figures
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.inventory
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.final_source_check
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.adaptive_recurrence_cut.validate
```

The profiler analysis reads the finalized native `cycle-*.tar.gz` bundles.
It does not use temporary unpacked files.
The eight rank traces preserve GPU events and CPU matrix shapes.

## Measurement limits

The initial map has eight requests per condition.
Its throughput is finite-cohort throughput, not sustainable serving capacity.
Correctness probes include heavy KV hashing and CPU copies.
Their elapsed times must not enter clean speedup tables.

The initial fixed-offset D-pressure profile does not itself prove that Decode was active before target arrival.
The `verified_D_pressure` profiles require two background requests with at least four output tokens each.
Those state-triggered probes are mechanism experiments, not Poisson serving traces.

The prepared long-generation manifests contain complete ShareGPT dialogue prompts.
Calibration and test each contain 200 disjoint prompt hashes.
Both model tokenizers satisfy the same length bins.
The common-support upper bound is 3,584 prompt tokens.
The controlled target output mean is 448 tokens.
That number is not an observed Natural-EOS output mean.
Manifest preparation does not mean that a serving experiment ran.
