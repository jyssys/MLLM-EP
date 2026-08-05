"""Create a fresh result directory, run one capture/replay, then analyze it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL = Path(
    "/home/esjung/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"
    "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    result_dir = args.result_dir or (
        repo
        / "poc_flashvep/results"
        / f"offline_wavefront_quick_poc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if result_dir.exists():
        raise FileExistsError(f"refusing to overwrite {result_dir}")
    result_dir.mkdir(parents=True)
    capture_path = result_dir / "layer24_capture.pt"
    request_path = result_dir / "capture_request.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo),
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "VLLM_NO_USAGE_STATS": "1",
            "FLASHVEP_PHYSICAL_GPUS": "4,5,6,7",
            "FLASHVEP_OFFLINE_RESULT_DIR": str(result_dir),
            "FLASHVEP_OFFLINE_CAPTURE_PATH": str(capture_path),
            "FLASHVEP_OFFLINE_MODEL_PATH": str(args.model_path),
            "FLASHVEP_OFFLINE_LAYER": "24",
            "FLASHVEP_OFFLINE_ORIGINAL_TOKENS": "799",
            "FLASHVEP_OFFLINE_VISION_TOKENS": "784",
            "FLASHVEP_OFFLINE_WARMUPS": str(args.warmups),
            "FLASHVEP_OFFLINE_ITERATIONS": str(args.iterations),
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "poc_flashvep/scripts/capture_offline_wavefront_input.py"),
            "--model-path",
            str(args.model_path),
            "--output",
            str(request_path),
            "--timeout-seconds",
            str(args.timeout_seconds),
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "poc_flashvep/scripts/analyze_offline_wavefront_quick_poc.py"),
            "--result-dir",
            str(result_dir),
            "--report",
            str(repo / "poc_flashvep/reports/offline_wavefront_quick_poc.md"),
            "--gate",
            str(
                repo
                / "poc_flashvep/results/baseline/gate_offline_wavefront_quick_poc.json"
            ),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        check=True,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
