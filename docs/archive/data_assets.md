# Phase 1 Assets and lmms-eval Dry Run

Checked on 2026-06-24.

## Model Snapshot

Downloaded to `models/Qwen3-VL-30B-A3B-Instruct`.

- Repo: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Files: 25
- Weight shards: 13 safetensors
- Local size: 58G
- Config/index/tokenizer/preprocessor files are present with the weights.

## Calibration Data

ShareGPT4V metadata:

- Repo: `Lin-Chen/ShareGPT4V`
- Metadata file: `data/sharegpt4v_meta/sharegpt4v_instruct_gpt4-vision_cap100k.json`

Materialized calibration subset:

- Manifest: `data/sharegpt4v_512/manifest.jsonl`
- Images: `data/sharegpt4v_512/images/`
- Count: 512 manifest rows, 512 image files, 0 zero-byte files
- Local size: 83M

Note: the public ShareGPT4V HF metadata contains source image paths such as
`coco/train2017/000000000009.jpg`, not embedded image binaries. The 512-image
subset was materialized from COCO train2017 URLs.

Reproduce:

```bash
python3 scripts/download_sharegpt4v_512.py
```

## Benchmark Data

Downloaded to `data/benchmarks/`.

| Benchmark | HF repo | Local size |
| --- | --- | ---: |
| MMMU | `lmms-lab/MMMU` | 3.2G |
| MMBench | `lmms-lab/MMBench` | 503M |
| ChartQA | `lmms-lab/ChartQA` | 70M |
| TextVQA | `lmms-lab/TextVQA` | 7.6G |
| MMStar | `Lin-Chen/MMStar` | 93M |

Reproduce model and benchmark snapshots:

```bash
python3 scripts/download_phase1_assets.py
```

## lmms-eval Status

Installed package:

- `lmms-eval 0.7.1`
- `deepspeed 0.19.2` imports on CPU; no DeepSpeed execution was run.

The PyPI wheel currently has a task-packaging issue: extensionless YAML include
templates are missing from the installed wheel, so `lmms-eval tasks list` fails
from site-packages. This matches the upstream GitHub issue "pip install ships
task configs without their extensionless include files" opened on 2026-05-21.

Workaround used for dry-run:

- Source zip downloaded to `external/lmms-eval`
- `PYTHONPATH=external/lmms-eval` lets task registry load the source-tree task files.

Dry-run command:

```bash
bash scripts/lmms_eval_dry_run.sh
```

Observed result:

- Command completed on CPU.
- Model: `dummy`
- Task: `mmmu_val`
- Limit: 1
- Mode: `--predict_only --log_samples`
- Output files:
  - `outputs/lmms_dry_run_mmmu/20260624_155054_results.json`
  - `outputs/lmms_dry_run_mmmu/20260624_155054_samples_mmmu_val.jsonl`

No real Qwen3-VL forward and no accuracy measurement were run.

