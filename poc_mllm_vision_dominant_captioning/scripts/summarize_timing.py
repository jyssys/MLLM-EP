from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from poc_mllm_vision_dominant_captioning.contract import atomic_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.results) if not row.get("is_warmup")]
    by_wave = defaultdict(list)
    for row in rows:
        by_wave[int(row["wave"])].append(row)
    walls = [values[0]["wall_ms"] for values in by_wave.values()]
    tpots = [row["mean_tpot_ms"] for row in rows if row.get("mean_tpot_ms") is not None]
    tpot_summary = {
        "mean_tpot_ms": float(np.mean(tpots)),
        "median_tpot_ms": float(np.median(tpots)),
        "p90_tpot_ms": float(np.quantile(tpots, 0.9)),
        "tpot_status": "MEASURED",
    } if tpots else {
        "mean_tpot_ms": None,
        "median_tpot_ms": None,
        "p90_tpot_ms": None,
        "tpot_status": "UNAVAILABLE_NO_PER_TOKEN_TIMESTAMPS",
    }
    summary = {
        "status": "MEASURED_CLEAN_TRUE_EP4",
        "global_batch": 8,
        "decode_length": 128,
        "measured_requests": len(rows),
        "measured_waves": len(walls),
        "fixed128_wave_e2e_ms_mean": float(np.mean(walls)),
        "fixed128_wave_e2e_ms_median": float(np.median(walls)),
        "tokens_per_second": float(len(rows) * 128 / (sum(walls) / 1000.0)),
        "trace_enabled": False,
        "speedup_claim": False,
    }
    summary.update(tpot_summary)
    atomic_json(args.output, summary)


if __name__ == "__main__":
    main()
