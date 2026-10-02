# Speculative decoding × physical MoE EP discovery

Source of truth: `poc_flashvep/reports/speculative_decoding_moe_ep_communication_discovery_spec.md` on the server (path/hash in local `artifacts/environment.json` and published `environment_public.json`). Only foreign user command paths are redacted in the public copy; numerical/model/topology evidence is unchanged. New branch is based on `dd73f832`, which contains the validated foundation/native EP infrastructure; the main dirty worktree is untouched.

Target Qwen3-30B-A3B `ad44e777…`, draft Qwen3-0.6B `c1899de2…`, BF16. Native vLLM greedy rejection sampling, correction/bonus, and KV rollback. No routing/expert-budget approximation or new scheduler.

Only `CUDA_VISIBLE_DEVICES=2,3,6,7`; TRUE TP2/DP2/EP4/PP1, DeepEP HT, linear32, EPLB/DBO/prefix cache off. Exactly one user request on DP0; DP1 participates using runtime dummy rows. Dummy/padding work is explicitly separated from user logical traffic, not called replicated requests.

## Reproduction

Run from the repository root. The GPU Python is the existing validated venv; CPU analysis uses the existing Anaconda base (pyarrow/sklearn/pytest), without modifying either environment.

```bash
# First acquisition/audit only; rejects an already-frozen cohort.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.audit

# Versioned bounded smoke; use a NEW --name to preserve earlier evidence.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.launch \
  --name my_gamma2_smoke --gamma 2 --lane trace --requests 2 --tokens 16 --warmup 3

# Initial discovery gate: seeded first16 (8 GSM8K + 8 HumanEval), cap64/EOS.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.campaign \
  --prefix phase_a_v1 --requests 16 --tokens 64

# Resume skips only successful immutable arms; failed IDs remain preserved.
# Clean timing expansion is permitted only after the Phase-A gate.
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.campaign \
  --prefix my_clean_v1 --lane clean --requests 2 --repeats 10 --tokens 64

/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.analyze \
  --version phase_a_v1 phase_a_v1_g0 phase_a_v1_g2 phase_a_v1_g4 phase_a_v1_g8
/home/esjung/anaconda3/bin/python -m pytest -q poc_speculative_moe_ep/tests
```

The completed result is in [reports/final_summary.md](reports/final_summary.md),
with primary clean labels under `artifacts/phase_b_v2/`. The96-request pool
(64 GSM8K/32 HumanEval) is frozen, but only its first16 (8+8) were used in the
bounded discovery gate. All SD outputs match AR. Physical rank features did
not improve clean grouped-CV error; no tested-depth EP crossover or practical
target-side oracle headroom was established. The remaining80 were **not**
silently marked evaluated. No new online method was implemented.

Exact completed physical commands (choose new prefixes when rerunning):

```bash
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.campaign \
  --prefix phase_b_clean_v1 --lane clean --requests 16 --repeats 3 --tokens 64
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.campaign \
  --prefix phase_b_key_reverse_v1 --gammas 8,4,2,0 --lane clean --requests 2 --repeats 7 --tokens 64
/home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_speculative_moe_ep.scripts.finish_measurement_plan
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.physical_validation \
  --version phase_b_v2 phase_b_clean_v1_g0 phase_b_clean_v1_g2 phase_b_clean_v1_g4 phase_b_clean_v1_g8
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.layer_matched --version phase_b_v2
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.verify_bundle --evidence compact_phase_a_v1
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.verify_bundle --evidence compact_phase_b_v1
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.verify_bundle --evidence compact_validation_v1
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.offline_reproduce --version phase_b_v2
```

`finish_measurement_plan` follows `artifacts/physical_plan.json`: it waits
boundedly for the reverse block, then runs bypass3-repeat arms and gamma2/8
route-repeat checks. Source changes affected only deferred audit metadata
during later blocks; no model/router math or scheduler semantics changed.

The initial provisional trace model and a corrected clean analysis are kept
versioned. `phase_b_v1` (local-only) incorrectly included the unbeatable
measured aggregate-truth oracle in the promotion comparator; `phase_b_v2`
corrects this to strongest fixed/expert/acceptance control and retains the
aggregate-truth oracle as a ceiling diagnostic. The same original timings
are used. Failed/interim runs remain locally preserved, not published as
headline measurements.

Per-invocation canonical ledger is
`phase_b_v2/verification_metrics_draft_aligned.parquet`. Native drafting after
frame t generates candidates for frame t+1; the original table's `draft_ms`
was a next-draft event. The append-only companion separates current `draft_ms`
and `next_draft_ms`. EP/target labels, features and all model/oracle conclusions
are unchanged; `draft_alignment.json` hashes both tables. Future analyzer
commands perform this association directly.

Each launch persists the exact manifest, raw stdout/stderr, per-request outputs and watchdog status. An outer bounded GPU transaction stops only the matching user-owned burner, rejects remaining competing PIDs, kills only owned experiment process groups on failure, and restores the original script with **explicit** `VLLM_UTILIZE_GPUS=2,3,6,7`. The script's server default is currently 0,1,4,5, so never invoke it without the override for this PoC.

## Evidence boundaries

- Trace lane: compact actual routes and deferred CUDA events; no hidden dumps or profiler. Trace timing is diagnostic until overhead/clean validation.
- Clean lane: no routes, minimal CUDA events and deferred sampler IDs; one drain RPC after the completed wave, not per layer.
- Bypass: identical driver with timing/route collection disabled; lightweight common wrappers remain. This is not a completely unpatched worker, and overhead checks are a small separate-restart subset.
- MoE components use **one cumulative-critical rank** per invocation; forward uses one maximum complete forward envelope. No component-wise maxima are added across ranks.
- Dispatch/combine intervals are full prepare/finalize composites (count/layout/transport/reduction), not pure network bandwidth.
- Logical bytes count BF16 hidden rows deduplicated by destination, not expert-path-count × hidden bytes. No claim about wire headers, actual HBM bytes, or independently removable rejected-row time.
- Target forward includes final norm/head and ends before native sampling. Draft cost and request wall are separately logged. Native request completion and DP-wave-drain wall are distinct quantities; old smoke wave-wall measurements must not be presented as request E2E.
- M0/M1/M2 are explanatory post-verification ridge models, not deployable online predictors. Same requests never cross CV folds; layers/ranks are not independent statistical samples.
- Gamma counts draft candidates; target input rows are generally gamma+1 (previous committed root plus candidates), including a correction or bonus output. Actual candidate count and output-cap truncation are recorded, not silently equated to nominal gamma.
- Offline oracle is not implemented/promoted without a passing missing-cost gate and same-prefix alternative alignment.

Full weights and dataset answer keys are not published. `raw/` is local-only; final compact raw evidence is selectively exported under `artifacts/`.
