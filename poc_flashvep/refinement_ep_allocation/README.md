# Refinement-aware expert allocation

This campaign uses `inclusionAI/LLaDA2.0-mini`, not `mini-preview`.
The model revision is `dad945cac317da394b390f82c7b40691d8a881ed`.
The current implementation covers the runtime audit and baseline evaluation.
Policy acceleration remains conditional on the specification gates.

## Current execution

The durable supervisor evaluates the HF baseline and physical EP4 baseline.
It saves each completed request before the next request.
It restores owned burners when a GPU group becomes free.
It does not approve later research phases automatically.
The separate baseline follow-up controller waits for that queue to finish.
It scores HumanEval and executes isolated P0/no-op A/B comparisons.
It does not implement approximate expert policies.

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.refinement_ep_allocation.prepare
python -m poc_flashvep.refinement_ep_allocation.audit
python -m poc_flashvep.refinement_ep_allocation.campaign start
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

The current quality queue uses Group A for EP4.
It uses four independent HF workers on Group B.
This concurrent execution does not provide isolated latency measurements.
Group B EP4 validation and A/B timing crossover remain pending.

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
The state file is `state/campaign.json`.
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
The supervisor lock rejects concurrent controllers.
Also check `state/followup.json` before another follow-up start.
The existing follow-up controller uses its own exclusive lock.
Completed request artifacts survive a worker failure.
The current queue does not retry a failed worker silently.
Review the failure log before a resume.

The user stopped the previous Looped MoE research.
Do not restart that research from this campaign.

## Provenance limitation

The initial driver source snapshot was not captured at process launch.
Do not identify the later Git checkpoint as the exact initial in-memory driver.
The model, reference helper, observer, and native runtime source hashes are available.
The driver corrections affect logging and observation, not model arithmetic or generation settings.
New worker starts save exact driver source snapshots and metadata.
