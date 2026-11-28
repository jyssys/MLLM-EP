# Modality-aware idle execution in MLLM EPD

The campaign asks whether a narrower task can use Decode resources to improve TTFT in native Dense MLLM EPD while preserving Decode isolation. It uses three physical H200s, the original Qwen2.5-VL-7B-Instruct checkpoint and native BF16 inference. The search starts with actual D service and transfer states, tests legally independent work on the original DAG, and promotes only candidates that show physical benefit and safe Decode behavior. [Candidate contracts](mllm_epd_autonomous_candidate_discovery.md) · [Original execution DAG](mllm_epd_execution_dag_and_resource_analysis.md) · [Prior mechanisms and examples](mllm_epd_idle_execution_prior_work.md)

## Runtime and assets

| Item | Recorded contract |
| --- | --- |
| Serving source | Worktree `/home1/irteam/MLLM_PD/h200-idle-execution`, branch `flashvep/mllm-epd-modality-aware-idle-execution-discovery`, based on `f5d6d70d750ea7b0a2c9f0599e874c07d4a5409b`. The launcher freezes source copies and hashes per epoch. |
| Launcher | `poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery.runner` reuses the predecessor's native buffered launcher and changes its worker extension. The opt-in vision RPC is local to the owned workers. |
| Hardware | Eight NVIDIA H200s behind NVSwitch/NVLink. Driver `580.126.16`. The actual experiment uses indices E=4, P=5 and D=6. The topology receipt records NV18 connectivity between them and NUMA node 1. Worker API endpoints use loopback 18840, 18841 and 18842. Native proxy uses 18847. NIXL side-channel ports are 19841 and 19842 with the role map reserving 19840 for E. |
| Serving software | `/home1/irteam/MLLM_PD/mllm_epd_runtime_env/bin/python`, Python 3.14.6, PyTorch `2.11.0+cu129`, vLLM `0.26.0+cu129`, NIXL `1.4.1`, Transformers `5.16.1`. vLLM official wheel SHA-256 `6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b`. |
| Assets | Sibling checkout `MLLM-EP-mllm-epd` at `449c3e1873c65206435a6bb9411a889a62e55be7`. The launcher verifies a bootstrap receipt, frozen originals, paths and file hashes. Model snapshot is `mllm_epd_autosetup/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5` below the same workspace. |
| Native model contract | Original BF16 weights, image and text, original prompt and media order, MRoPE, exact attention semantics. P/D TP1, B8192/S128, 42 GiB KV per LM worker, paged KV with block size 128, chunked prefill, native CUDA graphs, FlashAttention 3, CUDA IPC for EC and NIXL for KV. The maximum runtime sequence length is 32,768, including generation. |
| Cache and isolation | APC and worker multimodal processor cache are off. Requests use unique media UUIDs per occurrence. Request-local encoded output and page ownership remain necessary. No vision recomputation is credited as an EC hit. |

The lease verifies physical UUIDs and process ownership on 4/5/6. Restore the approved burner state after releasing the research workers and keep the receipts. Other GPU owners can remain present on the host, so CPU, host-memory and fabric activity must accompany an isolation claim. [Runtime pin and asset verifier](../mllm_epd_h200_prefill_heavy_batch_matched/runtime.py) · [Baseline server settings](../mllm_epd_prefill_heavy_token_split/servers.py) · [Owned campaign launcher](../mllm_epd_modality_aware_idle_execution_discovery/runner.py)

## Workload and comparison contract

- Use the frozen natural Very-long N24 roster with Fixed16 and open-loop Poisson QPS 1.465, seed 2026112809. Longer N64/N128/N256 trace lengths cycle the same roster with distinct prompt occurrences and frozen scheduled arrival offsets. QPS 0.879 supplies a lower-load control. Preserve actual media SHA-256, processor inputs and prompt-token receipts. Use the independent natural short-response EOS cohort for confirmation and distinguish its historical short-output selection from newly observed EOS length.
- Pair experimental arms with fresh Native at the same GPU assignment, budget, runtime, arrival trace, warmup, full prompt digest, output contract and transport. Keep APC and processor reuse settings, actual batch/graph shapes and KV reservations comparable. Compare optimized generic D-local whole-request/chunked routing and same-budget DP3 where measurements are available. DP3 uses three complete model replicas and its own prefill tuning. Historical or differently tuned DP3 is a context control unless the matched contract holds.
- Classify each scheduled attempt through completion, error or missing receipt. Start TTFT at scheduled arrival so client dispatch lateness counts. Obtain TPOT from time between first and last tokens per request, divided by generated gaps, and pool actual token gaps for ITL tails. Compute E2E and drain-based throughput across the finite cohort. One-token responses have TPOT zero and no gap samples. A finite overloaded cohort does not establish stable throughput. Report separate queue growth and a central observation window for stability claims.
- Use identical clean arms for the performance verdict. Make instrumented capture between clean before/after repeats with the same roster and warmup. Qualify only the measured observation and scope when the observer guard is within 5% and trace coverage, clock alignment, event drop checks and output integrity support it. D graph capture covers only CUDA graph launch envelopes. Within or outside those envelopes, eager paths, other processes and unobserved copies remain unclassified. [Workload freeze and figure inputs](mllm_epd_idle_execution_workloads_and_vision_geometry.md) · [Metric, pairing and completion reducer](../mllm_epd_modality_aware_idle_execution_discovery/online_evidence.py) · [CUPTI D graph coverage](../mllm_epd_modality_aware_idle_execution_discovery/graph_observer.py)

## Promotion gates

| Gate | Test and evidence |
| --- | --- |
| Native validity | E produces real original visual embeddings, P consumes them and generates real prompt KV, D acquires that KV and emits output on physical GPUs 4/5/6. Use ownership, native IPC and NIXL receipts and an eligible clean arm. |
| Motivation and usable intervals | In one aligned observation window, relate P compute-ready work and E readiness to D kernel, engine and transfer states. The preceding Prefill-heavy test used P busy ≥80%, P service ≥85% of its independent reference and a valid Decode SLO, with D busy ≤30% targeted. Encoder-busy identification additionally targeted E busy ≥80%. Keep proxy-quiet, CUDA-graph coverage, actual kernel idle and completed auxiliary work distinct. Replay may use the next native event only as a scoring label. |
| Physical effectiveness | Run real vision weights and tensors on the assigned GPU with all copies, completion, ownership and joins. Compare Native E, D whole-image, matched E-only shape/frontier, actual E+D and paired live native Decode. Measure device elapsed time and RPC end-to-end separately. Phase overlap reconstructed from individual physical durations is an analytical bound. |
| Correctness | Preserve original tensors, sequence order, image grids, masks, MRoPE, image/placeholder alignment, namespaces and lifetimes. The vision probe checks output shape and dtype, reports bitwise equality, differing elements, max absolute error, relative L2 and cosine similarity. No numerical tolerance in that probe automatically promotes a nonidentical output. Same-cut E and Native A/A localize numerical variation, while teacher-token logits, original greedy outputs and Natural EOS task quality require their own full serving tests. |
| Strict online result | TTFT p95 must be at least 15% below the strongest eligible same-budget baseline. Relative to Native, TPOT p95 and ITL p99 must each be at most 1.05×. Per-request absolute Decode requires TPOT ≤33 ms and **every** ITL ≤34 ms. The reducer also requires the candidate to have at least 95% absolute Decode attainment and match or exceed Native's pass count. Joint goodput additionally requires TTFT ≤5,000 ms. Report TPOT and ITL limits individually, joint and Decode goodput, and all failures. |
| Robustness and contribution | Repeat independent restarts and retain cases with missed, rejected or late work. Seek at least 128 matched primary requests and confirm on independent real input. A valid online claim additionally needs a reproducible incremental effect against the strong generic mechanism under the same constraints. Keep failed or untested gates explicit. |

The physical intervention checks the D role on controlled input and may prewarm fixture input and metadata. Offline replay labels policy fit against the unchanged Native timeline. Only an actual candidate that serves the original requests can establish real TTFT, useful work performed under admission, sustained Decode protection or task quality. Encoder bitwise equality and correct output IDs in an unrelated native Decode probe test different conditions.

## Reproduction on the recorded host

Set the working directory and paths. The serving interpreter is pinned. For plots, use a CPU interpreter with Matplotlib and expose it as `PLOT_PYTHON`. The run directory is external to the checkout and each new epoch needs an unused directory. Keep the verified asset repository and receipt in place.

```bash
cd /home1/irteam/MLLM_PD/h200-idle-execution
workspace=/home1/irteam/MLLM_PD
raw="$workspace/h200_idle_execution_runs"
prior="$workspace/h200_epd_modality_runs"
runtime="$workspace/mllm_epd_runtime_env/bin/python"
module=poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery
study=poc_flashvep/mllm_epd_modality_aware_idle_execution_discovery
assets="$workspace/MLLM-EP-mllm-epd"
receipt="$workspace/mllm_epd_autosetup/download_ready_1795600931728841878.json"
cache="$workspace/h200_memory_asymmetry_runs/epd_runtime_cache"
model="$workspace/mllm_epd_autosetup/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
launch_native() {
  "$runtime" -m "$module.runner" --asset-repo "$assets" --receipt "$receipt" \
    --cache-root "$cache" --p-budget 8192 --d-budget 8192 --max-seqs 128 --kv-gib 42 "$@"
}

python -m "$module.cohorts" --prior-root "$prior" --out "$raw/repro-dataset" \
  --receipts "$raw/repro-cohort-receipts.json"
CUDA_VISIBLE_DEVICES='' "$runtime" -m "$module.fixtures" \
  --manifest "$raw/dataset/primary_very_long_n24.json" --model "$model" \
  --out "$raw/repro-fixtures"
```

The launch plans are the external `recovery_plan.json`, `physical_plan.json`, `interference_plan.json`, `graph_plan.json`, `persistence_plan.json`, `dp3_matched_plan.json`, `matched_control_plan.json` and `kernel_roi_plan.json`. Exact resolved plans are also saved per epoch under `configs/frozen_plan.json`. The commands below use those original manifests and named probes. When reconstructing assets, update artifact locations while preserving recorded logical content and verify receipts before running. Execute epochs sequentially on the shared role ports.

```bash
launch_native --plan "$raw/recovery_plan.json" --run-dir "$raw/repro-native" \
  --inline-reroute --wall-seconds 1800
MLLM_IDLE_FIXTURES="$raw/fixtures" launch_native --plan "$raw/physical_plan.json" \
  --run-dir "$raw/repro-physical" --wall-seconds 1400
MLLM_IDLE_FIXTURES="$raw/fixtures" launch_native --plan "$raw/interference_plan.json" \
  --run-dir "$raw/repro-interference" --inline-reroute --wall-seconds 1400
launch_native --plan "$raw/persistence_plan.json" --run-dir "$raw/repro-persistence" \
  --inline-reroute --wall-seconds 2700
launch_native --plan "$raw/dp3_matched_plan.json" --run-dir "$raw/repro-dp3" \
  --layout dp3 --p-budget 2048 --wall-seconds 1800
"$runtime" -m poc_flashvep.mllm_epd_prefill_heavy_modality_discovery.replay \
  --proxy "$raw/repro-native/raw/proxy_epd.jsonl" --tag fresh_native_n64 \
  --out "$raw/dataset/fresh_native_n64_e_ready.json"
launch_native --plan "$raw/matched_control_plan.json" --run-dir "$raw/repro-matched" \
  --inline-reroute --wall-seconds 3100
MLLM_IDLE_PROFILE_WINDOW_S=20,55 launch_native --plan "$raw/kernel_roi_plan.json" \
  --run-dir "$raw/repro-kernel-roi" --inline-reroute --p-state-only \
  --state-period 350 --wall-seconds 2600
```

The previous `MLLM_IDLE_GRAPH=1` replay enabled graph mode after warmup had already instantiated the graphs. D crashed inside CUDA graph launch, so that arm provides no graph measurements. The exact-kernel run above uses the existing collector on a bounded center interval and measures the actual common ROI. Each run writes `state/environment.json`, model and ownership receipts, frozen source, `configs` with arrival hashes, and `raw` request/stage/vision records. Reducers consume raw files by scope.

```bash
"${PLOT_PYTHON:?Set PLOT_PYTHON to a CPU interpreter with Matplotlib}" \
  -m "$module.summarize_physical" --epoch "$raw/vision-physical-retry" \
  --epoch "$raw/vision-interference" \
  --out "$raw/repro-physical-analysis"
"$PLOT_PYTHON" -m "$module.online_evidence" --raw-root "$raw" --prior-root "$prior" \
  --out "$raw/repro-online-analysis" --report "$raw/repro-online.md" --figures
python -m "$module.idle_feasibility" --raw-root "$prior" --fresh-root "$raw" \
  --fresh-evaluate long:native-recovery/fresh_native_n128 \
  --out "$raw/repro-idle-feasibility"
```

The verdict figure reads an explicitly reviewed JSON artifact containing `scope` and `candidates`. Give each candidate A–G a row with `candidate`, `mechanism`, `verdict`, `evidence` and `basis`. The generator renders those decisions and does not infer them from surviving rows.

```bash
python -m "$module.verdict_figure" --input "$VERDICT_JSON" \
  --out "$study/results/figures/idle_candidate_verdict_matrix.svg"
```

See generated [physical results](../mllm_epd_modality_aware_idle_execution_discovery/results/vision_physical_summary.json), [online pair receipts](../mllm_epd_modality_aware_idle_execution_discovery/results/online_evidence.json), [replay evidence](mllm_epd_idle_motivation_qualification.md) and [frozen processor geometry](mllm_epd_idle_execution_workloads_and_vision_geometry.md). Re-run reports should use a pinned analysis revision, the corresponding raw epochs and a fresh output root.
