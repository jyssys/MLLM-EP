# Initial inventory

Status: discovery in progress. No track has a validated serving gain.

The experiment uses a new branch and an isolated runtime clone.
The parent commit is `b12fedb68a0861034f3ca684358acccfd5f84b0e`.
The upstream RLT commit is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
The experiment preserves the previous PoC and its verdict.

## Verified inputs

| Model | Checkpoint revision | Fixed depth | Native support | Upstream PD support |
|---|---|---:|---|---|
| Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | 4 | GPU smoke completed | GPU smoke completed |
| Huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | 32 | GPU smoke completed | Explicit rejection in `PDEngine.__init__` |
| Dory1.0-4B-BF16 | `b5437f97df4b998a505f4cac5b23791cd3e9df94` | 2 | Official model plugin pending | No model adapter |

The table records upstream support at the initial audit.
The experiment subsequently implemented explicit Huginn and Dory adapters.
All three models now execute actual NIXL PD on distinct GPUs.
Each model passed a 100-prompt bounded internal PD/A/B parity pilot.
Official-reference sequence agreement remains incomplete. See `03_baseline_parity_and_quality.md`.

Huginn uses the checkpoint's default depth, R32.
The earlier R16 PoC does not supply current R32 performance evidence.
Dory uses the official BF16 checkpoint, not the MXFP8 checkpoint.

The node has eight H100 80GB GPUs with NVSwitch connectivity.
The experiment records physical UUIDs in `configs/hardware.json`.
The first NIXL operator probe completed 40 measured copies with exact byte checks.
The copy probe is not a serving result.

## Environment

The new Python environment is `/home/esjung/.venvs/looped-pd-abc`.
It reads the validated package paths without changing those environments.
It uses Python 3.14.6, Torch 2.11.0+cu129, Transformers 5.16.1, and FA4 4.0.0b30.
The experiment installed NIXL CU12 1.4.1 only in the new environment.
The metapackage also declares a CU13 dependency. The experiment does not claim a clean `pip check`.

Initial serving smokes include compilation. They have no performance-acceptance role.
Final comparisons require separate warmup, independent starts, and isolated measurements.
The active controller and its queue are recorded under `state/`.
Draft final-file names do not imply a completed campaign or an accepted GO decision.
