# Environment manifest

Recorded 2026-09-12 KST before task-owned launches.

| Physical index | UUID | Device | task use |
|---:|---|---|---|
| 0 | `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d` | NVIDIA H100 80GB HBM3 | yes |
| 1 | `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb` | NVIDIA H100 80GB HBM3 | yes |
| 2 | `GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e` | NVIDIA H100 80GB HBM3 | yes |
| 3 | `GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8` | NVIDIA H100 80GB HBM3 | yes |

Every pair among physical GPUs 0--3 reports `NV18`. All task launches used
`CUDA_VISIBLE_DEVICES=0,1,2,3`. No task command used GPUs 4--7 and no unrelated
process was terminated.

- Driver: 570.148.08
- CUDA reported by driver: 12.8
- PyTorch: 2.8.0+cu128
- Python environment: `/home/esjung/.venvs/llada2-flash-sglang-053`
- Config SHA-256:
  `ac35e9dc8313f49b2600da2a8b3ec31c53c01d22a5943ec26c8e21f8e208ab07`
