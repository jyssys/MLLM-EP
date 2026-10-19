# Native decode-context parallelism eligibility

## Scope and result

The installed native DCP flag cannot provide an additional TP2/TP4 baseline for these two checkpoints.
Its validation requires TP greater than the total KV-head count for non-MLA attention.
Both Base checkpoints have sixteen query heads and sixteen KV heads.
TP2 and TP4 fail that condition before execution.

This is `SOURCE_CONSTRAINT_REPRODUCED`, not a failed physical CP experiment.
No GPU, vLLM model, Torch import, or new engine executes in this check.
It does not establish that every exact context-parallel layout is impossible or slow.
It also does not evaluate the separate prefill-context-parallel path.

## Actual installed source

[ModelConfig.verify_with_parallel_config](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/model.py:1249) first excludes the MHA16 configurations:

```text
if DCP > 1 and not MLA:
    require TP > total_KV_heads
    require DCP <= TP // total_KV_heads
    require (total_query_heads // total_KV_heads) % DCP == 0
```

[ParallelConfig](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/parallel.py:340) states that DCP does not expand the process count.
Without PCP, DCP reuses TP ranks.
Thus, a nominal TP4/DCP4 configuration is not automatically sixteen physical GPUs.
That group fact does not override the model eligibility checks.

The native [OuroAttention constructor](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:118) assigns disjoint KV heads when sixteen heads exceed TP.
The inspected DCP attention path gathers queries across its group and merges partition outputs with LSE.
Its [implementation](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:1155) is not permission to change head ownership without a correct layout.
No expert Dispatch/Combine or MoE All-to-All exists in this dense-model audit.

## CPU source check

The [script](../scripts/native_dcp_eligibility_cpu.py) extracts the exact inspected DCP conditional through Python AST.
It retains that source block and its SHA256 in the [artifact](../artifacts/native_dcp_eligibility_cpu_v1.json).
It reads both pinned checkpoint configs and hashes their files.
It executes only the eligibility block with explicit head counts.
It does not instantiate the complete model configuration.

Ten target configurations pass or fail as expected.
DCP1 passes this block.
Every eligible TP-divisibility combination with DCP greater than one fails for the target MHA16 geometry.
Positive controls with one or two KV heads pass three supported cases.
A two-KV-head, TP4/DCP4 negative control fails its maximum-DCP constraint.
These controls prevent a blanket claim that native DCP never works.

The check uses no random or synthetic model weights.
Its synthetic head-count controls test the source condition only.
They supply no target-model performance or quality evidence.

## Consequence

No DCP GPU job follows this source rejection.
A different TP×CP ownership proposal must specify Q/K/V head placement, sequence ownership, communication, and the exact merge.
It must preserve depth-specific KV and the existing BF16 model.
Those requirements do not identify a new recurrent mechanism by themselves.
The completed native TP1/TP2/TP4 controls remain the measured baseline.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.native_dcp_eligibility_cpu --output artifacts/native_dcp_eligibility_cpu_v1.json
```

The script refuses to replace an existing result.
