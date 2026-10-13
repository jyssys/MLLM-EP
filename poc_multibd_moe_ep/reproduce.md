# Reproduction

## Resumed confirmation: explicit user hardware override

On 2026-10-13 the user reassigned this PoC to **physical 4/5/6/7**. The
contract now pins that mask and those UUIDs. The commands in the historical
sections below describe the original 2/3/6/7 calibration, not permission to
use those GPUs now. Do not pool the interrupted old-GPU held-out measurements
with the new session or reuse old absolute latency as a matched baseline.
`authoritative_spec.md` is an exact hash-matched copy of the user-supplied
contract, whose original GPU restriction is superseded only by that explicit
user override. The original file remains untouched.

Fresh all-R EP4 and attempted same-checkpoint TP4 measurements use the unchanged frozen
24 held-out problem IDs. The session guard stops/restores only the exact
same-user `4,5,6,7` burner, and other GPU users are left untouched.

```bash
export PYTHONPATH="$PWD:/home/esjung/external/Diffulex-multibd-poc:/home/esjung/external/multibd-poc-deps"
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/resume_gpu4567_session --timeout 10800 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_confirmation --backend both --session gpu4567_20261013
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.analyze_conversion --session gpu4567_20261013
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.validate_ep4 --session gpu4567_20261013
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.validate_loaded_weights --session gpu4567_20261013
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.evaluate_quality --session gpu4567_20261013
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.build_report --session gpu4567_20261013
```

Use a unique session/output identifier for any reproduction: an existing
case directory or plan is deliberately not overwritten. New tables and
plots reside under `results/<session>` and `figures/<session>`, preserving the
earlier diagnostics. To restore separately, the launcher still requires an
explicit `VLLM_UTILIZE_GPUS=4,5,6,7`; do not trust its default GPU set.
The complete layer table is losslessly stored as `ep_metrics.csv.gz` to avoid
GitHub's individual-file limit. The earlier partial plain table is preserved
as `ep_metrics.intermediate_before_full_sweep.csv`, not a headline result.

## Validity repair and final frozen session

The first GPU4/5/6/7 confirmation was stopped after a real acceptance-schedule
mismatch. It must not be used as a completed primary sweep. The final session
uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch operations, and
disables automatic uninitialized-memory poison filling (which conflicted with
DeepEP's asynchronously written scratch). These are identical for every R and
both EP4/TP4 arms; their runtime cost is included, not presented as a gain.
No model precision, native thresholds, routing or frozen problem IDs change.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 POC_MULTIBD_DETERMINISTIC=1 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/confirmation_gpu4567_det_session --timeout 10800 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_confirmation --backend both --session gpu4567_det_20261014
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.audit_parity --directory poc_multibd_moe_ep/artifacts/R3_ep4_determinism_no_fill_probe_gpu4567 --output poc_multibd_moe_ep/results/gpu4567_20261013/determinism_no_fill_audit.json
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.analyze_conversion --session gpu4567_det_20261014
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.validate_ep4 --session gpu4567_det_20261014
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.validate_loaded_weights --session gpu4567_det_20261014
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.evaluate_quality --session gpu4567_det_20261014
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.build_report --session gpu4567_det_20261014
```

Use `--session gpu4567_det_20261014` for the final analyzer, validators,
quality evaluator and report builder. The final diagnostic passed six
repeats/setting and trace; OFF/ON final tokens/NFE/acceptance also matched.
The scratch-fill timeout and original acceptance failure remain in git.

### Final scope: valid primary EP4, rejected optional TP control

All four final EP4 arms completed all 24 requests with exact same-R
output/NFE/acceptance parity. The optional native `TP4/EP1` control then
failed the same strict repetition test at R3, HumanEval/11: final tokens,
NFE and acceptance differed. The all-backend guard consequently exited
nonzero and restored the previous burner. Do not reinterpret that exit as
a failed primary EP4 experiment, and do not use the failed TP samples to
claim an EP-vs-TP difference. Its internal cause is unresolved; no new engine
or replacement kernel was introduced to rescue an optional control.

`validity.json` retains the global failure flags and exposes separate
`ep4_held_out_pass_all` / `tp4_held_out_pass_all`. The report builder requires
all successful primary case statuses and excludes any incomplete/parity-invalid
TP control. It preserves `full_sweep_complete=false` while recording
`primary_ep_sweep_complete=true`; this is a scope distinction, not a relaxed
correctness test. Final verdict is **NO-GO** for the tested regime.

The read-only artifact audit reproduces the TP counterexample (expected exit1):

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.audit_parity --directory poc_multibd_moe_ep/artifacts/R3_tp4_held_out_paired_gpu4567_det_20261014 --output poc_multibd_moe_ep/results/gpu4567_det_20261014/tp_control_parity_audit.json
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m unittest poc_multibd_moe_ep.tests.test_contract
```

For a new primary-only GPU reproduction, use unique output/session names:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 POC_MULTIBD_DETERMINISTIC=1 /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/repro_primary_ep_session --timeout 10800 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_confirmation --backend ep4 --session repro_primary_ep
```

The primary findings, all 18 contract answers and caveats are authored in
`results/gpu4567_det_20261014/interpretation.md`; rebuilding this session's
report automatically includes them. Every model/evaluation/trace failure
is retained, with the relevant validity stratum. CPU analysis uses an empty
CUDA mask and does not require stopping the restored burner.

Restored burner stdout is append-only operational output, not an ongoing
experiment. The guard moves its still-open inode to a unique directory under
`/home/esjung/external/multibd-burner-live-*` and retains a bounded snapshot
plus `burner_log_archive.json` in the evidence. This keeps the committed
worktree stable without restarting/stopping a burner or deleting any log data.

## Original calibration provenance (historical GPU 2/3/6/7)

The commands below are an execution ledger for the original runner. They are
not a second current reproduction recipe: the current contract requires the
explicit 4/5/6/7 override and session-qualified analysis above. In particular,
do not rerun the metadata-freeze command against existing frozen artifacts.

Use the pinned `mbd-lms` branch, not upstream main. All GPU commands below are
scoped to physical 2/3/6/7. The PoC does not modify the external checkout or
shared Python environment. The two additional Python dependencies live in a
separate `--target` directory.

```bash
git clone --single-branch --branch mbd-lms https://github.com/SJTU-DENG-Lab/Diffulex.git /home/esjung/external/Diffulex-multibd-poc
git -C /home/esjung/external/Diffulex-multibd-poc checkout 9d1a04e794351f18039b3234a10e8b07e15bcbbf
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m pip install --target /home/esjung/external/multibd-poc-deps xxhash==3.6.0 easydict==1.13
export PYTHONPATH="$PWD:/home/esjung/external/Diffulex-multibd-poc:/home/esjung/external/multibd-poc-deps"
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.inspect_multibd_runtime
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.probe_state_machine
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m unittest discover -s poc_multibd_moe_ep/tests -v
```

GPU watchdog examples (new output directories preserve previous failures):

```bash
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/repro_original --timeout 240 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.probe_physical_ep --output "$PWD/poc_multibd_moe_ep/artifacts/repro_original"
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/repro_local_ids --timeout 240 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.probe_physical_ep --repair-local-ids --output "$PWD/poc_multibd_moe_ep/artifacts/repro_local_ids"
```

The first command is expected to fail correctness on pinned unmodified
dispatcher semantics. The second is expected to preserve all paths. Neither
synthetic diagnostic is MultiBD performance evidence.

Download the exact checkpoint before a full-model probe:

```bash
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("SJTU-DENG-Lab/MBD-Code-LLaDA2-mini-16B", revision="83434493ea9f026cbc139d681d7ae04ffefe052f", max_workers=4))'
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/repro_R1 --timeout 600 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_sweep --R 1 --smoke --mode trace --max-tokens 256 --output "$PWD/poc_multibd_moe_ep/artifacts/repro_R1"
```

Only after real-generation/physical-EP correctness passes should the same
runner be used with `--R 2/3/4`, clean/timing/trace lanes and the frozen
calibration/held-out manifests. `--backend tp4` is a same-checkpoint eager
non-EP control, not an offline rank mapping. Quality requires non-truncated
completions and official EvalPlus evaluation in a resource-limited sandbox;
the 64/256-token smoke is not pass@1 evidence.

The watchdog stops only an exact same-user burner supervisor for mask
2/3/6/7, checks for competing GPU processes, kills only its own spawned
session on timeout/error, and restores the previous burner argv/environment.
The server launcher defaults to a different GPU set, so standalone restoration
must explicitly pass `VLLM_UTILIZE_GPUS=2,3,6,7`.

## Matched calibration / held-out campaigns

Do not start these GPU commands concurrently. Every process has an outer
watchdog, and each case has a bounded request/NFE/process timeout. The frozen
manifest is identical for all R values. The confirmation runner retains failed
directories and refuses to overwrite them.

```bash
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/calibration_session --timeout 7200 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_campaign --phase calibration
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/confirmation_ep4_session --timeout 10800 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_confirmation --backend ep4
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_guarded --output artifacts/confirmation_tp4_session --timeout 10800 -- /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.run_confirmation --backend tp4
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.analyze_conversion
```

Held-out observers are counterbalanced IN THE SAME loaded-model session.
Clean mode restores the original D/E/C functions, rather than recording events
and merely ignoring their times. All-rank probe start/flush barriers are outside
timed inference. This prevents async shared-memory RPC output from crossing
probe boundaries. Earlier pre-barrier failures remain in `artifacts/` and are
listed in `results/validity.json`.

## Official Plus quality in isolation

Install only additional CPU evaluator packages into the isolated target. Do
not upgrade Torch/CUDA/DeepEP in the shared environment.

```bash
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m pip install --target /home/esjung/external/multibd-poc-deps --no-deps evalplus==0.3.1 datasets==3.6.0 pyarrow==19.0.1 pandas==2.2.3 dill==0.3.8 multiprocess==0.70.16 wget==3.2 appdirs==1.4.4 tempdir==0.7.1 termcolor==2.5.0 pytz==2025.2 tzdata==2025.2 python-dateutil==2.9.0.post0 six==1.17.0
```

Quality requires Bubblewrap with working private user/mount/network/PID
namespaces. On this server the official Ubuntu package was downloaded and
extracted to `/home/esjung/external/multibd-sandbox/unpacked`; it was not installed
into the operating system. `evaluate_quality` uses official EvalPlus base AND
full Plus test inputs/oracles inside Bubblewrap with resource limits. There
are no host-side generated-code `exec` calls. The sandbox has no GPU or network
device and cannot access the rest of the user's home.

```bash
PYTHONPATH="$PWD:/home/esjung/external/multibd-poc-deps" /home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.evaluate_quality
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python -m poc_multibd_moe_ep.scripts.build_report
```

Normal EOS is not a truncated benchmark sample. Pinned upstream's
`is_truncated` property includes EOS clipping, so the evaluator checks the
actual output/context/NFE/repetition cap flags instead. The original mistaken
pre-audit diagnostic remains in `results/quality.json`; only
`results/quality_evalplus.json` contains official scores.
