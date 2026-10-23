# Implementation and clock scope

This project adds an objective plug-in to the existing legal RLT chunk controller.
It does not implement a new serving engine or model-parallel architecture.
Only independent single-GPU workers on physical GPUs4–7 run research jobs.
The paused native-vLLM Ouro Loop PP program remains paused.

## Reused components

- The pinned pretrained Base Ouro-1.4B weights and tokenizer.
- RLT model runner, attention/GEMM backend, paged KV, native events, streams, and CUDA graphs.
- The prior removable G2/G3 adaptation and legal frontier enumeration.
- Completion-aware fairness, one-prefill-span admission, bounded progress fallback, cost lookup, and numerical probes.
- Frozen full-answer quality prompts, natural stop handling, and preserved evaluator.

## New code

| File | Purpose | Changes model math? |
| --- | --- | --- |
| `objective_features.py` | Causal normalized E/R features and support flags | No |
| `objective_families.py` | O0–O13 score rules and deterministic tie resolution | No |
| `adaptive_selector.py` | Restrict the common legal universe to Joint, C-only, or G-only | No |
| `action_diagnostics.py` | Reproduce previous actual plans and classify C opportunities | No |
| `objective_search.py` | Historical TRAIN survey and fixed-C regret tuning | No |
| `select_final.py` | Freeze objective from fresh VALIDATION and fixed controls from TRAIN | No |
| `workload_generator.py` / `final_plan.py` | Source-family splits, natural inputs, and registered arrival plans | No |
| `work_ledger.py` | Native prompt submission accounting and safe callback teardown | No |
| `serving_eval.py` | Arrival/output clocks, raw evidence, and Full-R4 assertions | No |
| `quality.py` / `grade_quality.py` | Full-answer public pilot and question-paired scoring | No |
| `dispatch.py` / `resources.py` | Independent workers, resume checks, and owned burner handling | No |
| `audit_analysis.py` / `mechanism_analysis.py` | Explicitly scoped diagnostic tables | No |

No optimizer can resize a live staged chunk.
No optimizer can skip recurrent work or group decode rows into a prefill-only G action.
The common candidate universe determines projected features before a single-axis restriction.
This prevents a control from receiving weaker telemetry merely because its action surface is smaller.
The four live objective families use the same frozen service lookup, budget, fairness, and kernels.

## Serving measurement

Inputs are pretokenized. The client runs inside the model-worker process.
E2E begins at the registered arrival time and ends at native EOS or the registered chat-stop token.
The all-returned E2E column also contains caps. Complete-E2E columns exclude caps and label that denominator.
TTFT ends at the first delivered output token.
ITL uses consecutive delivered output timestamps.
TPOT is a request-level decode average, not one recurrent-step duration.
Queue delay and work triggered after arrival remain inside serving latency.

Model loading and input-only warm-up occur before the serving window.
Lazy graph capture during a request remains inside that request's latency.
This project does not measure an HTTP server, network delay, or tokenizer latency.
It does not claim production API capacity.

## Separate evidence lanes

The four-worker characterization lane maximizes experiment throughput.
The decisive serial lane reserves authorized GPUs4–7 and runs one measurement worker at a time.
Existing GPU0–3 background processes remain unchanged in both lanes.
Event-observer audits are separate from clean serving.
Native event waits precede inspected GPU state. No unfinished kernel buffer is read without synchronization.
Host submission descriptors are not GPU finish timestamps.

O0–O13 offline scores do not constitute alternative-action execution.
Constructed-state sensitivity is SIMULATION_ONLY.
Forced paired C interventions are component diagnostics, not adaptive full-answer gains.
Finite drained cohorts are not sustainable serving capacity.
