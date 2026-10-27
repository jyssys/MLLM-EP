# Refinement-aware expert allocation

This campaign uses `inclusionAI/LLaDA2.0-mini`, not `mini-preview`.
The model revision is `dad945cac317da394b390f82c7b40691d8a881ed`.
The implementation covers baseline evaluation, causal sensitivity diagnostics, structural replay, and physical operator checks.
The user reduced the active pilot to GSM8K 128 and HumanEval 32 questions.
The original full-evaluation queues are canceled.
The baseline pilot is complete.
The retained-all validation stage also completed 16 requests across two EP4 groups.
P1/P2/P3 each complete 16 free-running validation trajectories at two removal budgets.
Dense semantic intervention changes expert contributions, but preserves all physical expert work.
The separate native operator probe removes assignments before DeepEP.
It does not implement a sparse end-to-end generation engine.
Current verdict: `NO_GO` for the tested proxy design and native EP4 bridge.
The discovery queues are complete. No sparse generation-engine expansion is approved by their results.
See `../reports/11_final_decision.md` for the decision and missing evidence.

## Current execution

The pilot supervisor evaluates the HF baseline and physical EP4 baseline.
It saves each completed request before the next request.
It restores owned burners when a GPU group becomes free.
It does not approve later research phases automatically.
The pilot reuses completed GSM8K generations from both backends.
It generates and scores the selected HumanEval responses.
It also checks identity-hook parity on those native HumanEval requests.
The previous isolated-baseline follow-up no longer runs automatically.

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.refinement_ep_allocation.postprocess --pilot --score-humaneval
```

The preparation command uses the existing CPU Python environment.
That environment supplies PyArrow and PyYAML.
The HF environment has neither `datasets` nor PyArrow.
Do not install new packages into the validated model environments.

## Source dependencies

| Component | Pinned source |
|---|---|
| HF helper and sandbox | `MLLM-EP-mini-repro`, commit `c02563979fe818a93c3f6f64b162cdf3df69a8e2` |
| Physical EP4 bridge observer | `MLLM-EP-dllm-comm-amplification`, commit `dc539d3222193ab0902e0a342679edc3610f54bb` |
| dInfer checkout base | `external/dinfer-llada2-flash-poc`, commit `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` |
| Active dInfer modifications | Exact source hashes in `environment.json` |

The checkout base alone does not identify the patched DeepEP bridge.
Check the source hashes before a reproduction run.
The earlier EP4 repository contains its bridge patch.
No existing model source file changes in this campaign.

## Runtime configuration

Group A uses physical GPUs `0,1,2,3`.
Group B uses physical GPUs `4,5,6,7`.
Each EP4 process uses one logical CUDA device.
The native bridge combines dense TP4, source sequence sharding, and physical EP4.
It does not represent four full model replicas.

The original concurrent HF/EP4 quality queue ended.
The user-approved pilot preserves its completed GSM8K generations.
Both physical EP4 groups passed retained-all validation checks.
Sensitivity collection uses two independent EP4 replicas.
Isolated timing reserves the opposite group as quiet, without burners.
The user explicitly approved this temporary measurement exception.

The supervisor sets these variables for the validated native runtime:

```text
TORCHDYNAMO_DISABLE=1
SGLANG_ENABLE_JIT_DEEPGEMM=false
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
OMP_NUM_THREADS=4
```

The HF worker uses BF16 and SDPA.
The EP4 worker uses the existing BF16 Triton expert path and DeepEP normal mode.
It disables compilation and CUDA Graph capture for all current EP4 comparisons.
This is not a fully optimized production engine.

## Quality protocol

GSM8K uses the official dInfer Mini four-shot prompt examples.
HumanEval uses the original task prompt in the official chat template.
GSM8K generation permits 2,048 new tokens.
HumanEval generation permits 1,024 new tokens.
Both paths permit EOS and use block length 32 and threshold 0.95.

The HF path preserves the official implementation's multinomial call at temperature zero.
The native decoder uses its original greedy path.
These two paths are distinct evaluation protocols.
The published benchmark protocol is not proven identical to either local protocol.

Calibration contains 256 GSM8K training prompts without labels.
Validation contains 128 different training prompts.
Validation answers reside in a separate scoring artifact.
The online worker does not receive those answers.

## Evidence

Artifacts reside in `poc_flashvep/results/refinement_aware_ep_expert_allocation/`.
The pilot state file is `state/pilot.json`.
The trace collection state file is `state/pilot_traces.json`.
The command ledger is `commands.jsonl`.
The job ledger is `job_events.jsonl`.
The supervisor log is `logs/supervisor.log`.

```bash
python -m poc_flashvep.refinement_ep_allocation.resources status
/home/esjung/.venvs/llada2-flash-sglang-053/bin/python \
  -m poc_flashvep.refinement_ep_allocation.trace_analysis --run-id A_sanity
```

`ep_trace_audit.json` distinguishes logical payload estimates from physical traffic counters.
The trace analysis does not report an implemented sparse-execution speedup.
The two-request sanity test does not establish benchmark quality.

The reduced pilot quality table is `pilot/baseline_accuracy.csv`.
The validation summary is `pilot/refinement_trace_summary.json`.
The observer breakdown is `pilot/observer_breakdown.csv`.
These files do not establish an allocation policy's speedup.

Raw interrupted and validation runs also reside in bounded archive parts.
Each part is at most 80,000,000 bytes.
The archive index records all part hashes and the reassembled archive hash.
Source raw files remain unchanged.
The follow-up archive is `archives/sensitivity_and_physical_cost.index.json` under the result directory.
It contains 1,420 entries, including failed attempts and raw per-rank operator records.
Most bulky raw directories require archive extraction after a repository clone.
The summary CSV files and SVG figures remain directly accessible in Git.
The archive entry point is `python -m poc_flashvep.refinement_ep_allocation.archive_evidence`.

The native B1 return removes EOS tokens.
Postprocessing corrects the initial logger's stop labels without rewriting raw evidence.
The native legacy NFE counter also undercounts chunked prefill calls.
Postprocessing derives that correction from the audited prefill geometry.
New worker starts capture the terminal buffer and count model forwards directly.

## Measurement contract

Primary latency requires one isolated EP4 group.
The other group must remain quiet during that measurement.
Burners count as interference and must stop for this measurement.
Burners must resume after the isolated measurement.

The main accuracy test uses natural generation.
An exact output token budget is not a forced generation length.
HumanEval completions must run inside the existing namespace sandbox.
Do not execute generated code directly on the host.

The speedup estimator is baseline E2E divided by candidate E2E.
Report latency reduction separately from speedup.
Bootstrap paired requests, not individual tokens.
Do not sum overlapping stage times into a global critical path.

## Resume

Check the supervisor PID and current GPU owners before another start.
The pilot lock rejects concurrent pilot controllers.
The current state is `state/pilot.json`.
Do not restart the canceled full-evaluation controllers.
Completed request artifacts survive a worker failure.
The sensitivity queue skips complete jobs and can adopt a verified active worker.
It does not retry a failed worker silently.
Review the failure log before a resume.

The user stopped the previous Looped MoE research.
Do not restart that research from this campaign.

## Provenance limitation

The initial driver source snapshot was not captured at process launch.
Do not identify the later Git checkpoint as the exact initial in-memory driver.
The model, reference helper, observer, and native runtime source hashes are available.
The driver corrections affect logging and observation, not model arithmetic or generation settings.
New worker starts save exact driver source snapshots and metadata.

## Follow-up commands

Check active owners and durable state before a new start.
Do not start overlapping controllers for the same GPU groups.

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.refinement_ep_allocation.resources status
python -m poc_flashvep.refinement_ep_allocation.sensitivity_campaign start
python -m poc_flashvep.refinement_ep_allocation.cost_campaign start
python -m poc_flashvep.refinement_ep_allocation.cost_campaign start-clean
```

The clean follow-up waits for the baseline queue.
It measures operator wall time without the separate stage-event observer.
Each controller restores owned burners after its GPU work ends.
Completed run IDs do not execute again on resume.
An incomplete rank result set requires a new run ID and a recorded failure review.

Use these commands after the corresponding worker stages finish:

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-flash-sglang-053/bin/python \
  -m poc_flashvep.refinement_ep_allocation.sensitivity_analysis
python -m poc_flashvep.refinement_ep_allocation.allocation_replay
python -m poc_flashvep.refinement_ep_allocation.cost_analysis
python -m poc_flashvep.refinement_ep_allocation.figures
python -m poc_flashvep.refinement_ep_allocation.validate_cpu
```

The default CPU environment has NumPy and Matplotlib but no Torch.
The sensitivity scorer imports the frozen parser through its original Torch-dependent module.
Its command therefore uses the existing native environment with CUDA devices hidden.
It does not create a GPU inference job.

Important raw and derived paths:

| Path under the result directory | Meaning |
|---|---|
| `pilot/sensitivity_summary.csv` | Full-trajectory validation accuracy and NFE |
| `pilot/sensitivity_samples.jsonl` | Per-question answers, parity, transitions, and intervention counts |
| `pilot/allocation_replay.csv` | Matched P2–P5 structural counts, not latency |
| `pilot/isolated_baseline_summary.csv` | Isolated natural-generation P0 repeats |
| `pilot/operator_cost_summary.csv` | Physical operator wall timings and work counts |
| `latency_samples.csv` | Individual isolated baseline timing repetitions |
| `figures/` | Editable SVG figures |
| `state/sensitivity.json` | Durable semantic-screen state |
| `state/cost.json`, `state/cost_clean.json` | Isolated cost queues |

Stage-event intervals can include host submission gaps.
The separate clean operator lane does not contain those event probes.
Neither lane is a candidate request speedup.
The current evidence does not pass the incremental algorithm or EP-headroom gates.
