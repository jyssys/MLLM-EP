"""Analyze a completed bounded visual-streaming run (stdlib + NumPy only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vision, layers = [], []
    for path in sorted(root.glob("timing_worker_pid*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = row.pop("kind", None)
            if kind == "vision": vision.append(row)
            elif kind == "decoder": layers.append(row)
    return vision, layers


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _duration_table(root: Path, vision: list[dict[str, Any]], layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("driver_dp*.json")):
        obj = json.loads(path.read_text())
        if obj.get("ok"): records.extend(obj.get("records", []))
    vby: dict[str, list[float]] = {}
    for row in vision: vby.setdefault(str(row["active_id"]), []).append(float(row["duration_ms"]))
    lby: dict[str, list[dict[str, Any]]] = {}
    for row in layers: lby.setdefault(str(row["active_id"]), []).append(row)
    def stage_samples(rows: list[dict[str, Any]], stage: str) -> list[float]:
        """Recover one 48-layer sample per worker/repetition.

        Request ids stay stable across repetitions so the serving path is
        unchanged.  The ordered JSONL rows therefore contain one natural
        48-layer stack per repetition; summing the whole stream would mix a
        3x prefill with a 1x vision measurement and corrupt the oracle.
        """
        by_pid: dict[Any, list[dict[str, Any]]] = {}
        for item in rows:
            if item.get("stage") == stage:
                by_pid.setdefault(item.get("pid"), []).append(item)
        samples: list[float] = []
        for seq in by_pid.values():
            for start in range(0, len(seq) - 47, 48):
                block = seq[start:start + 48]
                if len(block) == 48:
                    samples.append(sum(float(x["duration_ms"]) for x in block))
        return samples

    out = []
    for row in records:
        active = str(row["request_id"]); ev = vby.get(active, [])
        lr = lby.get(active, []); pre = [float(x["duration_ms"]) for x in lr if x.get("stage") == "prefill"]
        dec = [float(x["duration_ms"]) for x in lr if x.get("stage") == "decode"]
        item = dict(row); item["vision_encode_ms"] = float(np.median(ev)) if ev else np.nan
        # A decoder layer is observed once per TP/DP worker.  Sum each
        # complete layer stack separately, then take the median replica and
        # repetition; never sum all repetitions together.
        pre_samples = stage_samples(lr, "prefill")
        dec_samples = stage_samples(lr, "decode")
        item["prefill_layer_cuda_ms"] = float(np.median(pre_samples)) if pre_samples else np.nan
        item["decode_layer_cuda_ms"] = float(np.median(dec_samples)) if dec_samples else np.nan
        item["prefill_sample_count"] = len(pre_samples)
        item["decode_sample_count"] = len(dec_samples)
        item["lm_plus_decode_ms"] = float(item["wall_ms"] - item["vision_encode_ms"]) if ev else np.nan
        out.append(item)
    return out


def _oracle(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Records are repeated for measurement repetitions.  Collapse each
    # logical request to its median before constructing the image dependency
    # graph; repetitions are not additional images.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(str(item["request_id"]), []).append(item)
    by: dict[str, dict[str, Any]] = {}
    for key, items in grouped.items():
        item = dict(items[0])
        for field in ("wall_ms", "vision_encode_ms", "prefill_layer_cuda_ms", "decode_layer_cuda_ms"):
            vals = [float(x[field]) for x in items if np.isfinite(float(x.get(field, np.nan)))]
            if vals:
                item[field] = float(np.median(vals))
        by[key] = item
    rows, reasons = [], []
    for n in (2, 4):
        enc = [float(by[f"independent_{i}"]["vision_encode_ms"]) if f"independent_{i}" in by else np.nan for i in range(1, n + 1)]
        pref = []
        for i in range(1, n + 1):
            cur = by.get(f"prefix_{i}"); prev = by.get(f"prefix_{i-1}") if i > 1 else None
            # Prefix segment cost is the difference of measured cumulative
            # LM prefill CUDA work.  Keep negative differences visible rather
            # than clipping them: a negative increment means this bounded
            # trace is too noisy for a causal oracle claim.
            pref.append(float(cur["prefill_layer_cuda_ms"] - (prev["prefill_layer_cuda_ms"] if prev else 0.0)) if cur else np.nan)
        multi = by.get(f"multi_{n}")
        if multi is None or not all(np.isfinite(enc)) or not all(np.isfinite(pref)) or any(x < 0 for x in pref):
            reasons.append(f"n={n}: missing/negative segment timing"); continue
        enc_clock = pref_ready = 0.0; timeline = []
        for i, (e, p) in enumerate(zip(enc, pref), 1):
            enc_start = enc_clock; enc_clock += e; p_start = max(enc_clock, pref_ready); pref_ready = p_start + p
            timeline.append({"image": i, "encode_start_ms": enc_start, "encode_end_ms": enc_clock,
                             "prefill_start_ms": p_start, "prefill_end_ms": pref_ready,
                             "encode_ms": e, "prefix_segment_ms": p})
        serial = sum(enc) + sum(pref); critical = pref_ready; measured = float(multi["wall_ms"])
        rows.append({"image_count": n, "measured_baseline_wall_ms": measured,
                     "decomposed_serial_ms": serial, "oracle_streaming_ms": critical,
                     "oracle_reduction_vs_decomposed": (serial - critical) / serial,
                     "oracle_reduction_vs_measured_wall": (measured - critical) / measured,
                     "oracle_speedup_vs_decomposed": serial / critical,
                     "encode_sum_ms": sum(enc), "prefill_sum_ms": sum(pref), "timeline": timeline})
    return rows, {"status": "COMPUTED" if rows else "NOT_COMPUTABLE", "reasons": reasons,
                  "rows": [{k: v for k, v in x.items() if k != "timeline"} for x in rows]}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True); args = ap.parse_args(); root = args.result
    manifest = json.loads((root / "workload_manifest.json").read_text())
    vision, layers = _read_jsonl(root)
    _write_csv(root / "vision_timing.csv", vision); _write_csv(root / "decoder_layer_timing.csv", layers)
    eq_csv = root / "equivalence_results.csv"
    if eq_csv.exists() and (root / "equivalence_summary.json").exists():
        eq_summary = json.loads((root / "equivalence_summary.json").read_text())
    else:
        eq_summary = {"status": "NOT_AVAILABLE", "comparisons": 0, "ok_comparisons": 0,
                      "min_cosine": None, "max_abs_error": None,
                      "tolerance": "No precomputed equivalence file"}
        eq_csv.write_text("status,reason\nNOT_AVAILABLE,no precomputed equivalence file\n")
    records = _duration_table(root, vision, layers); _write_csv(root / "timing_results.csv", records)
    oracle_rows, oracle_summary = _oracle(records)
    (root / "oracle_results.json").write_text(json.dumps(oracle_summary, indent=2) + "\n")
    timeline_rows = []
    for row in oracle_rows: timeline_rows.extend([{**x, "image_count": row["image_count"]} for x in row["timeline"]])
    _write_csv(root / "oracle_timeline.csv", timeline_rows)
    if oracle_rows:
        plt.figure(figsize=(8, 4.5))
        for row in oracle_rows:
            x = [0] + [z["image"] for z in row["timeline"]]; y = [0.0] + [z["encode_end_ms"] for z in row["timeline"]]
            plt.plot(x, y, marker="o", label=f"{row['image_count']} images")
        plt.xlabel("completed image encodes"); plt.ylabel("encoder elapsed (ms)"); plt.title("Dependency-aware streaming oracle"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(root / "timeline_streaming_oracle.png", dpi=180); plt.close()
        pos = np.arange(len(oracle_rows)); labels = [str(x["image_count"]) for x in oracle_rows]; w=.35
        plt.figure(figsize=(7, 4.5)); plt.bar(pos-w/2, [x["measured_baseline_wall_ms"] for x in oracle_rows], w, label="measured baseline"); plt.bar(pos+w/2, [x["oracle_streaming_ms"] for x in oracle_rows], w, label="ideal streaming")
        plt.xticks(pos, labels); plt.xlabel("image count"); plt.ylabel("ms"); plt.title("Baseline vs ideal streaming"); plt.grid(axis="y", alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(root / "baseline_vs_oracle.png", dpi=180); plt.close()
    else:
        # Required figures are still emitted, explicitly showing that the
        # measured prefix increments were not a valid causal oracle input.
        for name, title in (("timeline_streaming_oracle.png", "Streaming oracle unavailable"),
                            ("baseline_vs_oracle.png", "Baseline measured; oracle unavailable")):
            plt.figure(figsize=(8, 4.5)); plt.text(.5, .5, "NOT COMPUTABLE\nnegative/noisy prefix increment", ha="center", va="center", fontsize=13)
            plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(root / name, dpi=180); plt.close()
    reductions = [float(x["oracle_reduction_vs_decomposed"]) for x in oracle_rows]
    if eq_summary.get("status") == "PASS" and reductions and min(reductions) >= .15: status = "STRONG_GO"
    elif eq_summary.get("status") in ("PASS", "APPROXIMATE") and reductions and min(reductions) >= .10: status = "GO"
    elif eq_summary.get("status") in ("PASS", "APPROXIMATE") and reductions and min(reductions) >= .05: status = "HOLD"
    else: status = "NO_GO"
    gate = {"FINAL_STATUS": status, "IMAGE_LEVEL_EQUIVALENCE": eq_summary.get("status", "NOT_AVAILABLE"),
            "REAL_PIPELINE_PROTOTYPE": "NOT_RUN", "GPU_EXECUTION_STATUS": "COMPLETED",
            "configuration": manifest.get("configuration", {}), "equivalence": eq_summary,
            "oracle": oracle_summary, "lm_separated_embedding_equivalence": "NOT_DIRECTLY_EXPOSED_BY_CURRENT_VLLM_API"}
    (root / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n"); print(json.dumps(gate, indent=2))


if __name__ == "__main__": main()
