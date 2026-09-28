from __future__ import annotations

import argparse
from pathlib import Path

from poc_mllm_vision_dominant_captioning.contract import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keep = (
        "request_id", "source_request_id", "pair_id", "condition", "repeat_id",
        "domain", "analysis_split", "topology", "dp_rank", "wave",
        "image_sha256", "image_size_rgb", "image_grid_thw", "vision_token_count",
        "prompt_token_count", "universal_prompt_sha256", "output_text",
        "output_token_ids", "output_token_count", "finish_reason", "stop_reason",
        "forced_token_exact", "forced_token_sha256", "ttft_ms", "mean_tpot_ms",
        "wall_ms", "evidence", "is_warmup",
    )
    rows = [
        {key: row.get(key) for key in keep if key in row}
        for row in read_jsonl(args.input)
    ]
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
