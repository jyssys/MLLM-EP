# Reproduction

Run commands from `/home/esjung/MLLM-EP-github`.
Use the pinned environment in `environment_versions.json`.
No model download or new package installation is required on the measured host.

## Source preparation

The upstream RLT revision is `4a350344027960a48384626aa646f1765fda6d4f`.
Create a separate worktree at `/home/esjung/external/ouro-elb-rlt` from that revision.
Apply `poc_flashvep/ouro_elb/patches/grouped_decode.patch` inside that worktree.
Do not apply the older prefill grouping patch.
The model snapshot is `/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

```bash
git -C /home/esjung/external/ouro-budget-joint-rlt worktree add --detach /home/esjung/external/ouro-elb-rlt 4a350344027960a48384626aa646f1765fda6d4f
git -C /home/esjung/external/ouro-elb-rlt apply /home/esjung/MLLM-EP-github/poc_flashvep/ouro_elb/patches/grouped_decode.patch
```

These commands create a new path. Do not run them over an existing measured worktree.
For a new host, clone the official RLT repository first and use the same pinned revision.

## Software invariants

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/esjung/external/ouro-elb-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m unittest poc_flashvep.ouro_elb.test_leases -v
```

The software fixtures do not measure pretrained task quality.
`native_regression.xml` retains the separate upstream CPU regression results.

## Registered queues

```bash
/home/esjung/anaconda3/bin/python -u -m poc_flashvep.ouro_elb.dispatch --manifest poc_flashvep/reports/ouro_8gpu_elastic_loop_batching_decode_fusion_discovery/p1_v2_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -u -m poc_flashvep.ouro_elb.dispatch --manifest poc_flashvep/reports/ouro_8gpu_elastic_loop_batching_decode_fusion_discovery/p2_screen_v2_plan.json --manage-burners
/home/esjung/anaconda3/bin/python -u -m poc_flashvep.ouro_elb.dispatch --manifest poc_flashvep/reports/ouro_8gpu_elastic_loop_batching_decode_fusion_discovery/contention_plan.json --manage-burners
```

`p1_crossed_plan.json` completes the missing async/sync and graph combinations.
`lifecycle_plan.json` tests cancellation, ID reuse, and genuinely executed heterogeneous frontiers.
`intervention_v2_plan.json` warms all candidate row widths before the paired component action.
`barrier_plan.json` separates initial prefill fragmentation from steady decode grouping.
`quiet_plan.json` confirms selected cells with three process starts on each of two GPUs.
`profile_plan.json` adds a separate CUDA-event observer lane.
`mixed_baseline_plan.json` measures native Mixed D1 without an ELB policy.
`nsight_plan.json` records separate heavy CUPTI traces when supported.
`mixed_groups_plan.json` crosses D1/D2/D4 inside the native Mixed path.
`ablation_plan.json` crosses graph on/off and sync/async on the same three GPUs.
`natural_stop_v2_plan.json` contains the reviewed natural chat-stop lifecycle probes.
The old natural-stop plan remains a preserved failed attempt.

The dispatcher masks each process to exactly one approved physical device.
It records UUID, UID, affinity, source text, runtime diff, and job identity.
Jobs retain independent contexts, logs, and cache paths.
Every GPU crosses D1/D2/D4 rather than receiving one policy permanently.
The dispatcher assigns the next queued job automatically.

Completed immutable artifact hashes permit safe resume without rerunning finished jobs.
Failed or interrupted attempts require reviewed new IDs.
Never delete old attempts to force resume.
Worker completion excludes mutable dispatcher telemetry.
The final raw evidence manifest hashes telemetry after process exit.

## Analysis and ownership

```bash
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_elb.analyze
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_elb.analyze_nsight
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_elb.figures
taskset -c 128-143 /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_elb.validate
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_elb.resources status
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_elb.resources restore 0,1,2,3,4,5,6,7
```

The resource manager signals only matching UID/session/log-owned burner processes.
An unrelated GPU job causes refusal rather than termination.
Declared quiet controls reserve all eight devices without burners during measurement.
Other unused authorized devices retain owned burners.
The final handoff restores all eight burners.

`profiles/nsight_raw_bundle.tar.gz` contains both complete Nsight reports and both SQLite exports.
Its SHA256 is `9a2319f42320e579b3a5d4289e0fb9238473edc549f2bec1211a21b217cfcf89`.
The original files remain local. The compressed bundle and portable trace CSVs support repository access.

No TP/PP engine or paused native-vLLM Loop PP task belongs to these commands.
