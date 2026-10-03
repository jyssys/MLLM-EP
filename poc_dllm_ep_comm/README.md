# Physical dLLM EP communication amplification

Characterization only: baseline native LLaDA2-mini denoising, real four-rank
DeepEP sparse dispatch/combine, deferred routing summaries, offline accounting.
No acceptance predictor, expert pruning, migration, Epoch port, or custom kernel.

Authoritative spec: [frozen copy](references/authoritative_spec.md).
Output directory is independent and append-only. The source repository's dirty
index is not used; this branch is an independent git worktree.

## Existing environment

- Model `inclusionAI/LLaDA2.0-mini`, revision
  `dad945cac317da394b390f82c7b40691d8a881ed` (cached local snapshot).
- Python `/home/esjung/.venvs/llada2-flash-sglang-053/bin/python`.
- dInfer `/home/esjung/external/dinfer-llada2-flash-poc`, upstream
  `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`, plus the exact pre-existing
  [bridge patch](patches/dinfer_existing_bridge.diff). Do not apply it to an
  already-dirty checkout. For a fresh checkout at that SHA, use
  `git apply /path/to/dinfer_existing_bridge.diff` and verify the source hashes
  recorded in `results/environment.json`.
- BF16, 20 layers / 19 sparse layers, E256 / K8 / D2048 / M512; 64 routed experts
  physically owned per EP rank. One shared expert per sparse layer is not an EP
  routed path and is explicitly excluded from routed-byte accounting.
- Dense TP4 + source-SP4 + physical EP4 + DP1 + PP1. This is not Qwen's TP2-DP2
  topology. A single user request's rows are partitioned once across four source
  ranks, not duplicated as four DP requests.
- `CUDA_VISIBLE_DEVICES=2,3,6,7`; logical ranks 0,1,2,3 map to those physical IDs.

## Commands

From repository root:

```bash
/home/esjung/anaconda3/bin/python -m poc_dllm_ep_comm.scripts.prepare
/home/esjung/anaconda3/bin/python -m pytest -q poc_dllm_ep_comm/tests
/home/esjung/anaconda3/bin/python -m poc_dllm_ep_comm.scripts.launch --name smoke_new --limit 2
/home/esjung/anaconda3/bin/python -m poc_dllm_ep_comm.scripts.launch --name main_new --limit 32 --secondary
/home/esjung/anaconda3/bin/python -m poc_dllm_ep_comm.scripts.analyze --run main_new --output main_new
/home/esjung/anaconda3/bin/python -m poc_dllm_ep_comm.scripts.figures --result main_new
```

The committed frozen manifest must not be regenerated/retuned after routing
observations. For exact inputs, reuse it. `bash .../scripts/reproduce.sh new_name`
runs the existing frozen manifest. Every run name must be new.

Natural B1 generation uses official mini block32/threshold0.9/prefix-BD defaults,
temperature0 and native EOS stop. Generation budget64 is rounded by native
dInfer to the next block boundary (record actual output lengths). Secondary
block64 uses the first four frozen problems per domain, unchanged threshold.
Request watchdog120s, wave watchdog2400s, no infinite retries.

Final `main_v2` additionally sets `TORCHDYNAMO_DISABLE=1` consistently across
all arms after a recorded SGLang TorchInductor failure. Native Triton expert
GEMMs and DeepEP transport remain enabled. This is a compiler workaround, not a
new kernel or decoding algorithm. The exact error and bounded amendment are in
`raw/main_v1/status.json` and `results/runtime_amendment.json`.

The GPU transaction stops only the exact authorized-mask user burner, verifies
no competing GPU process remains, and restores it in `finally`. The burn script
must be called with `VLLM_UTILIZE_GPUS=2,3,6,7`; its unqualified default targets
different GPUs and must not be used here.

## Evidence formats

`raw/RUN/REQUEST_b32/trace.rankR.json` contains model call/iteration IDs, exact
acceptance positions and final token state. Matching compressed NPZ contains
global position IDs, top-8 IDs/weights, mask state before/after, and actual
received owner-local expert IDs. No hidden-state or expert-output dump.
`timing.rankR.json` is an independent event-only pass; `bypass.result.rankR.json`
is clean E2E. All four ranks' ownership tensors and receive buffer sizes are
audited. Decoder control/output identity is checked across observer modes.

Logical remote payload is one 4096-byte BF16 row per unique remote destination
rank in Dispatch and one corresponding partial-output row in Combine. It is
not top-k paths times hidden bytes, not transport packet/NVLink wire counters,
and excludes route/control metadata and dense TP all-gather/all-reduce traffic.
Local paths are reported but are not remote bytes. Receive buffers are checked
against all destinations (including local), not just remote destinations.

`B_live_only` and `B_one_shot` are offline structural views, never measured
speedups. Prompt/context is separately decomposed from already-accepted output.
EOS/tail sensitivity is computed using final-returned positions only as well as
all materialized accepted positions. Gate promotion uses the conservative
final-returned-position residual, not a favorable EOS artifact.

The same cumulative critical rank supplies Dispatch, Expert and Combine sums;
components from unrelated per-stage rank maxima are never added. Communication
share of whole request wall is a conservative denoising lower bound; model-only
forward share is a separately labeled diagnostic. Heavy trace wall is not a
headline serving latency. Missing tuned Triton configs and eager execution are
documented limitations, not speedup evidence.
