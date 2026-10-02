# Agentic Serving × Physical MoE EP: cheap discovery

The authoritative spec is the server's
`poc_flashvep/reports/agentic_moe_ep_poc_spec.md`; a byte-exact snapshot is
published with the evidence. This work is an independent PoC on branch
`flashvep/agentic-moe-ep-locality-poc`, based on validated commit `d25b0495`.
No original dirty/staged work or prior PoC result was modified.

Result: [report.md](report.md). Statistically detectable adjacent locality
survives shared-prefix exclusion, but previous-turn routing is worse than
coding-domain global popularity. **NO-GO for the tested agent-aware premise**;
no learned predictor, scheduler, migration or new kernel was built.

## Frozen contract

- Target Qwen3-30B-A3B, revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`;
  BF16,48 MoE layers,128 experts,K8,d2048,m768.
- Physical `2,3,6,7` only, logical `0,1,2,3`; TRUE TP2-DP2-EP4-PP1, DeepEP HT,
  sparse Triton, linear32, EPLB/redundant/DBO off. Prefix caching **on**;
  eager, chunked prefill off, one active request on DP0, DP1 runtime dummy.
- Twelve instance-disjoint public SWE-smith trajectories, first9 semantic
  turns each;4 calibration,8 held-out,64 adjacent held-out pairs.
- No routing/outcome/answer-based sample selection. Context preserved up to
  frozen16384-token envelope; no live agent tools/commands executed.
- Primary only NEW PREFILL SUFFIX. The previous assistant's repeated semantic
  history is additionally excluded using a current-template observation
  boundary, because Qwen3 can change empty-think formatting across turns.
- Secondary first32 recorded assistant tokens, teacher-forced in causal
  packed prefill; not natural AR quality/latency.
- Global uses calibration only; previous uses completed prior turn only;
  oracle and non-adjacent diagnostic controls may use future traces only
  under explicitly nondeployable labels. No causal policy sees next truth.
- Trajectory bootstrap2000, eight independent held-out trajectory units;
  not64 pairs or48 layers as independent samples. CI conditional on fixed
  four-trajectory calibration, not a claim of cross-repository generalization.

## Actual reproducible commands

Run from `/home/esjung/MLLM-EP-agentic-ep` (or repository checkout root).
Existing GPU environment and separate existing CPU analysis interpreter are
reused; nothing was upgraded or reinstalled.

```bash
# Acquisition freezes metadata-selected public semantic trajectories.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_agentic_ep.scripts.collect_or_reconstruct_trajectories
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_agentic_ep.scripts.audit

# Append-only stronger exclusion metadata; cohort/input IDs unchanged.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_agentic_ep.scripts.freeze_exclusion_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.freeze_controls

# Actual completed run. Choose a NEW name to rerun; no overwrite/resume duplication.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_agentic_ep.scripts.launch \
  --name main_v2 --timeout 1800

/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.aggregate \
  --run main_v2 --version main_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.analyze_locality --version main_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.simulate_preparation --version main_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.finalize --version main_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.export_evidence \
  --run main_v2 --version main_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.verify_evidence --version main_v2
/home/esjung/anaconda3/bin/python -m pytest -q poc_agentic_ep/tests
```

Frozen manifests already exist, so acquisition/freeze commands correctly
reject an existing cohort. GPU launches accept a new `--name`; successful
sample checkpoints are durable in local `raw/`. Failed run IDs/results are
preserved and require a new ID, never unbounded retry. One outer1800-second
transaction and per-wave watchdog bound GPU use. Analysis output versions
must also be new, except the read-only verifier.

The historical `smoke_v1` collected two calibration turns before detecting
the template-history rewrite. Original `frozen_manifest.json`, partial
`results/smoke_v1`, and `smoke_validation_v2` remain locally preserved.
`frozen_manifest_v2.json` supersedes exclusion metadata only, with cause and
original hash. No held-out routing was observed before that repair. Mainv2
is the sole authoritative hypothesis evaluation.

## Published evidence / offline reproduction

`results/main_v2/raw_routing/*.npz` contains new-only actual expert IDs,
original FP32 weights, source ranks, input positions and phase flags.
`destination_ep_ranks.npz` is the explicit destination companion under the
verified actual owner map. Counts are in `aggregated.npz`; raw and normalized
derived metrics appear in layer/turn/pair/EP CSVs and JSON.

Compact semantic event files retain actual DeepEP backend/loaded-shape proof,
source-SP partition, actual KV-cache query starts and D/E/C CUDA events.
`native_worker_records.json` preserves sample/cache/checkpoint order.
Instrumented timings are **diagnostic**, not headline serving latency.

`oracle_policy_assignments.json.gz` publishes all simulated replica choices;
the larger plain JSON stays local. Only the explicit oracle uses next-turn
truth. Replication is assignment-optimal top-frequency under per-source slots,
not a globally byte-optimal transport oracle. Logical bytes preserve
destination dedup; no assignment-to-E2E conversion is made. All48 layers
count toward replica memory:9MiB/expert/layer implies0.421875GiB per extra
slot/rank across the model. Idle sources do not prepare zero-score experts.

`p2p.json` has actual12-direction9MiB/64MiB CUDA-copy samples (warmup3/repeat10).
Tool-gap extrapolation is serial-copy feasibility only: not concurrent
migration, allocator/layout overhead, or real tool latency. Max-rank compute
load can worsen after local replication; report explicitly shows that risk.

No GPU is required for these published-evidence checks/reanalysis:

```bash
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.verify_evidence --version main_v2
/home/esjung/anaconda3/bin/python -m poc_agentic_ep.scripts.offline_reproduce --version main_v2
```

## GPU ownership / cleanup

Existing bounded GPU transaction checks owner/PID/UUID, stops only this
user's authorized-mask burner, rejects competing processes, and cleans only
owned experiment process groups. Every success/failure/exception restores
the original `/home/esjung/vllm-ep/run_utilize.sh` with explicit
`VLLM_UTILIZE_GPUS=2,3,6,7`. The script currently defaults to0,1,4,5, so bare
invocation would violate this PoC. Final four-GPU burner state is recorded.
Other GPUs/processes are never modified.
