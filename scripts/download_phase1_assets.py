"""Download Phase 1 model and benchmark assets.

This script is intentionally download-only. It does not run model forward,
accuracy evaluation, or GPU-dependent code.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download


MODEL_REPO = "Qwen/Qwen3-VL-30B-A3B-Instruct"
BENCHMARK_REPOS = {
    "MMMU": "lmms-lab/MMMU",
    "MMBench": "lmms-lab/MMBench",
    "ChartQA": "lmms-lab/ChartQA",
    "TextVQA": "lmms-lab/TextVQA",
    "MMStar": "Lin-Chen/MMStar",
}


def main() -> None:
    model_path = snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        local_dir="models/Qwen3-VL-30B-A3B-Instruct",
        max_workers=8,
    )
    print(f"model: {model_path}")

    for name, repo_id in BENCHMARK_REPOS.items():
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=f"data/benchmarks/{name}",
            max_workers=8,
        )
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

