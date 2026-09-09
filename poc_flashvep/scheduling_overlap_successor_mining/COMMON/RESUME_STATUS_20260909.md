# Resumption checkpoint — 2026-09-09 10:18 KST

## 10:27 KST — external blocker cleared

User confirmed GPUs released. Fresh query confirms 4–7 empty, with unrelated
jobs only on 0/1 untouched. NanoFlow numerical screen actually started under
`numerical_resume_20260909_b4_retry1`; original blocked directory is preserved.
Native Layered Prefill installation also completed successfully. Resume the
planned equal screening. Sections below describe the earlier temporary blocker.

Study: Layered Prefill × FastPP × NanoFlow, unchanged branch and results root.
User explicitly reauthorized GPU 4,5,6,7 and idle/final burn on that subset.

## Actual progress

- Read the complete working spec and preserved deferred list.
- FlashAttention editable build from the pause completed successfully.
- Started remaining native Layered installation with CUDA_VISIBLE_DEVICES="",
  fixed SM90 compilation target and four compiler jobs. CPU build only; no
  automatic GPU import or benchmark is queued after pip completion.
- Attempted the prepared NanoFlow same-prefix numerical screen. Its occupancy
  guard stopped before any native worker/model measurement began.
- Fresh live GPU experiment time on this resumption: **0 minutes**. This is an
  external resource blocker, not a baseline failure or a negative research result.

## Occupancy race / OS permission blocker

Initial 10:13 check showed GPUs 4–7 empty. Other-user jobs subsequently started.
Read-only PID/device mapping confirmed these exact processes only used 4–7:

| Owner | PID | Physical GPUs | Role |
|---|---:|---|---|
| kaist3 | 3997233 | 4,5,6,7 | retrieval_server.py |
| kaist3 | 3999947 | 4 | stepsearch worker |
| kaist3 | 3999949 | 5 | stepsearch worker |
| kaist3 | 3999954 | 6 | stepsearch worker |
| kaist3 | 3999934 | 7 | stepsearch worker |

The user authorized termination on these GPUs, but all five exact-PID SIGTERM
attempts returned `PermissionError: [Errno 1] Operation not permitted` under
the current `esjung` account. `sudo -n -l` returned `a password is required`.
No password was requested or bypass attempted. No process was terminated.
No processes/GPU jobs on 0–3 were touched.

Owner/admin release of GPUs 4–7 is needed before clean measurements. Do not
launch experiments or burn alongside these jobs or label occupied but 0%-util
GPUs as free. Do not kill broad process groups or unrelated parent workflows.
PID lists are historical evidence; re-resolve live targets before any action.

## Burn scope

The local `/home/esjung/vllm-ep/run_utilize.sh` defaults to `1,2,3,4` despite
the user's expectation. Explicit `CUDA_VISIBLE_DEVICES=4,5,6,7` and
`VLLM_UTILIZE_GPUS=4,5,6,7` are required whenever burn is safely launched.
No burn is running for this study while GPUs are occupied by other users.

## Files / sessions

- CPU build log: `raw/layered_native_resume_20260909.log` under the existing
  `scheduling_overlap_successor_mining_20260908_161510` results root.
- NanoFlow attempted result directory: `nanoflow_runs/numerical_resume_20260909_b4`.
  Created before occupancy assertion; no experiment result exists there. Use a
  new run name on retry rather than overwriting it or calling it completed.
- Paused old build wrapper remains stopped; do not blindly SIGCONT it.
- No GPU experiment supervisor remains running and no automatic retry is queued.
