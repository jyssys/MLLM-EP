# Source and prior-work audit

## Archive access and scope

The connected [MLLM-EP experiment archive](https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489) became available after the API server restart.
The read-only query returned 35 relevant rows.
The archive describes 183 reports from 165 branches as of 2026-11-10.
The query does not cover every archived report.
The targeted search did not return the current publication parent `6cadf1011a0b596efe39ccccefc9898dd8ef2925`.
The campaign uses that Git commit and its original reports for the Recovery baseline.

The archive records three relevant historical findings:

| Archived result | Check against original source | Scope for this campaign |
| --- | --- | --- |
| Adaptive idle-time intervention | Huginn moves 5/384 requests. Loop/Native TTFT p95 is 1.47201 [1.09531, 1.98167]. Goodput ratio is 0.31264 [0.12379, 0.50631]. [Original report](https://github.com/jyssys/MLLM-EP/blob/0adb7491c866d6d66cf71ea30dca63f82c46d135/poc_flashvep/reports/ouro_huginn_adaptive_idle_intervention_dynamic_loop_final_summary.md) | These results use the old execution policy and Native denominator. They do not measure physical idle capacity. |
| Deep static validation | Huginn Loop31/Native TTFT p95 is 0.9800 [0.8373, 1.1471], n=3. [Original report](https://github.com/jyssys/MLLM-EP/blob/034b4ef9d2a8f0fe1a9d993370693d349cd03838/poc_flashvep/reports/ouro_huginn_loop_specific_native_split_deep_validation_final_summary.md) | The interval does not confirm a static benefit. The report separates that result from structural logical-layer equivalence. |
| D-local shared-core batching | A stronger same-depth control also captures the batching gain. The report gives `NO_GO_LOOP_BATCHING`. [Original report](https://github.com/jyssys/MLLM-EP/blob/3b3633dbc64bcd916c2ba1ff513deb4226cc647d/poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_final_summary.md) | This decision covers incremental batching. It does not measure the current continuation policy or a clean kernel gap. |

The local Git objects match the linked revisions.
The source receipt records the archive query and three page reads.
Raw workspace responses remain local. The publication excludes their contents.
The campaign does not change the archive.

## External serving systems

**FluidPD.** The paper uses pressure estimates on both sides of PD.
FluidToken can transfer a computed Token prefix and its KV while D completes the suffix.
FluidRole changes instance roles and has a different deployment scope.
This campaign retains a fixed 1P1D deployment and uses a comparable Token suffix arm.
It does not claim a full PPI/DPI controller, FluidRole, or official FluidPD code.
The inspected source did not establish an available official implementation.
[FluidPD paper](https://arxiv.org/html/2610.06917v1).

**DuetServe.** Its main mechanism multiplexes Prefill and Decode on spatial GPU resources.
It also prepares Decode work in advance.
The Qwen3-8B 8000/200 PD motivation differs from the main multiplexing experiment.
The campaign's optional stock AR diagnostic uses that workload geometry only.
It is not a reproduction of DuetServe's implementation.
The paper's CUDA, driver, and Runtime revisions differ from this campaign.
[DuetServe paper](https://proceedings.mlr.press/v306/gao26m.html).

**Prefill Deflection.** The method can place a new request on D before P performs its Prefill.
That placement can avoid a mid-request transfer.
This campaign instead resumes an exact recurrent frontier after P executes part of the prompt.
The transfer, lifetime, and completion costs therefore differ.
[Prefill Deflection paper](https://arxiv.org/html/2607.02043v1).

**SGLang and vLLM.** The inspected SGLang Decode path separates KV preallocation, transfer readiness, and execution.
The campaign retains these distinctions in its Native reservation and ownership audit.
The vLLM NIXL path supports the separate stock AR diagnostic.
The Huginn implementation uses the qualified private vLLM-RLT Runtime.
[SGLang Decode source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/decode.py).
[vLLM disaggregated serving documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/).

**Local LoopServe reference.** The reviewed source separates the Runtime owner, loop planning, and graph execution.
The new source pipeline retains one owner thread and uses existing graph completion events.
It does not copy the reference's cache layout or execution ABI.
A GPU-resident R32 control path remains unqualified in the Native Runtime.

## What transfers to Huginn

Huginn's fixed-depth contract permits explicit grouping without early exit.
The campaign keeps every recurrent iteration and removes only unused Gate readback.
The qualified numerical fix still uses fixed 256-row Down and adapter projection tiles.
Equal semantic recurrent rows therefore do not establish equal launches, padded work, or execution time.

The standard AR diagnostic and the Huginn campaign use different KV layouts and model work.
Huginn stores valid recurrent KV at each depth.
Its physical pool also reserves nonrecurrent layer slots in those depth blocks.
Wire bytes, reserved quota, and GPU HBM allocation are distinct quantities.

A logical-layer implementation can reproduce a Loop cut if it uses the same operations, aliases, and state handoff.
This is `EQUIVALENT_MECHANISM`.
No separate logical-layer performance result follows from that equivalence.
A measured advantage over one Token baseline does not prove unique recurrent architecture value.
