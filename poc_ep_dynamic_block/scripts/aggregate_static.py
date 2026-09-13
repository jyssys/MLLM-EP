#!/usr/bin/env python3
"""Aggregate clean variable-block runs without treating observer traces as timing."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG_RE = re.compile(
    r"(?P<regime>[^_]+)_(?P<task>gsm8k|humaneval)_n(?P<n>\d+)_"
    r"B(?P<block>\d+)_m(?P<mini>\d+)_g(?P<gen>\d+)_t(?P<threshold>[0-9.]+)"
    r"(?:_L(?P<target_total>\d+))?_r(?P<repeat>\d+)"
)
FORWARD_RE = re.compile(
    r"^Forward:\s*(?P<nfe>\d+),\s*Time:\s*(?P<time>[0-9.]+),.*?"
    r"TPS:\s*(?P<tps>[0-9.]+).*?TPF:\s*(?P<tpf>[0-9.]+)", re.MULTILINE
)
EFFECTIVE_THRESHOLD_RE = re.compile(r"\bthreshold=(?P<value>[0-9.]+)")


def decode_many(path: Path) -> list[dict]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    offset = 0
    values = []
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        values.append(value)
    return values


def main() -> None:
    rows = []
    for log in sorted((ROOT / "logs").glob("*.log")):
        match = TAG_RE.fullmatch(log.stem)
        if not match or match["regime"] == "trace":
            continue
        log_text = log.read_text(errors="replace")
        timing = FORWARD_RE.search(log_text)
        if not timing:
            continue
        effective_match = EFFECTIVE_THRESHOLD_RE.search(log_text)
        requested_threshold = float(match["threshold"])
        effective_threshold = (
            float(effective_match["value"])
            if effective_match else requested_threshold
        )
        run_dir = ROOT / "results" / "clean" / log.stem
        quality_path = run_dir / "quality.json"
        output_files = list(run_dir.glob("*.jsonl"))
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
        outputs = decode_many(output_files[0]) if output_files else []
        gpu_path = ROOT / "logs" / f"{log.stem}_gpu.csv"
        peak_hbm = None
        mean_util = None
        if gpu_path.exists():
            hbm, util = [], []
            with gpu_path.open(newline="") as handle:
                for record in csv.reader(handle):
                    try:
                        hbm.append(float(record[3]))
                        util.append(float(record[5]))
                    except (IndexError, ValueError):
                        pass
            peak_hbm = max(hbm) if hbm else None
            mean_util = sum(util) / len(util) if util else None
        rows.append(
            {
                **match.groupdict(),
                "threshold": effective_threshold,
                "threshold_requested": requested_threshold,
                "threshold_effective": effective_threshold,
                "nfe": int(timing["nfe"]),
                "bct_s": float(timing["time"]),
                "tps": float(timing["tps"]),
                "tpf": float(timing["tpf"]),
                "quality_correct": quality.get("correct"),
                "quality_total": quality.get("total"),
                "accuracy": (
                    quality.get("correct", 0) / quality["total"]
                    if quality.get("total")
                    else None
                ),
                "generated_tokens": sum(int(x.get("generated_length", 0)) for x in outputs),
                "peak_hbm_mib": peak_hbm,
                "mean_sampled_gpu_util_pct": mean_util,
                "clean_timing": True,
                "run_dir": str(run_dir.relative_to(ROOT.parent)),
            }
        )
    fieldnames = list(rows[0]) if rows else []
    with (ROOT / "STATIC_BLOCK_SWEEP.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
