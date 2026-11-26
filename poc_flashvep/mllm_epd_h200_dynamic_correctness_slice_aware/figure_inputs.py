"""Select published figure series from the campaign's underlying receipts."""

import argparse
import csv
import json
from pathlib import Path


def correctness(path):
    results = []
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            case = row["fixture"].removeprefix("serial_E_lead_").replace("_", " ")
            count = int(row["requests"])
            if not count:
                continue
            results.append({
                "label": f"{row['regime'].upper()} {row['transport']}\n{case}",
                "final_kv_byte_equal_percent": 100 * int(row["suffix_KV_byte_equal_layers"]) / (28 * count),
                "greedy_exact_percent": 100 * int(row["exact_greedy"]) / count,
            })
    return {"evidence": "DIAGNOSTIC_AND_CORRECTNESS_PASS", "values": results}


def funnel(paths):
    keys = (
        ("internal_chunk", "Observed internal chunk"),
        ("candidate_geometry", "Feasible geometry"),
        ("candidate_and_no_later_inflight", "Feasible without hold"),
        ("callback", "Decision callback"),
        ("callback_and_D_idle", "Callback with D idle"),
        ("old_supported_cost", "Cost support observed"),
        ("old_predicted_fit", "Budget fit"),
        ("reservation", "Reservation acquired"),
        ("actual_offload", "Actual suffix"),
        ("actual_offload_and_decode_SLO", "Actual suffix and Decode SLO"),
    )
    values = {"stages": ["Main100 roster", *[label for _, label in keys]], "counts": {}}
    for path in paths:
        data = json.loads(path.read_text())
        full = data["full"]
        source = data["source"]
        counts = [full["requests"]]
        for key, _ in keys:
            if source == "stock" and key not in ("internal_chunk", "candidate_geometry",
                                                   "candidate_and_no_later_inflight"):
                counts.append(None)
            else:
                counts.append(full["stages"][key]["requests"])
        values["counts"][data["run"]] = counts
    values["scope"] = "Stock observations have no decision attempts. Blank positions indicate unavailable stages."
    return {"evidence": "DIAGNOSTIC", "values": values}


def stock_timeline(path):
    with path.open(newline="") as source:
        candidates = [
            row for row in csv.DictReader(source)
            if row["central_ROI"] == "True" and row["candidate_geometry"] == "True"
            and float(row["observed_D_HTTP_gap_ms"]) > 0
            and float(row["D_engine_queue_quiet_gap_ms"] or 0) > 0
        ]
    if not candidates:
        return {"evidence": "NOT_MEASURED"}
    row = min(candidates, key=lambda candidate: float(candidate["time"]))
    ready = float(row["P_own_native_ready_after_cut_ms"])
    events = {
        "P chunk issue → worker result": [[-float(row["issue_to_result_ms"]), 0]],
        "D host queue quiet": [[0, float(row["observed_D_HTTP_gap_ms"])]],
        "D engine queue quiet": [[0, float(row["D_engine_queue_quiet_gap_ms"])]],
        "Source P-ready": [[ready, ready]],
    }
    if row["P_next_other_ready_after_cut_ms"]:
        ready = float(row["P_next_other_ready_after_cut_ms"])
        events["Next other P-ready"] = [[ready, ready]]
    return {
        "evidence": "DIAGNOSTIC",
        "values": {
            "label": "First central Native cut with both D queues quiet",
            "request_id": row["dataset_request_id"],
            "later_p_tokens_inflight": int(row["later_tokens_inflight"]),
            "events": events,
        },
    }


def forecast(path, model_path):
    from ..mllm_epd_h200_modality_aware_dynamic_split.policy import consume_budget
    from .slack_policy import predict, ready_features

    model = json.loads(model_path.read_text())
    discovery = set(model["training_requests"])
    series = {"Old 80K / 160 ms": [], "Observed-progress estimate": []}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if (row["drain"] == "True" or row["dataset_request_id"] in discovery
                    or not row["P_next_other_ready_after_cut_ms"] or not row["progress"]):
                continue
            progress = json.loads(row["progress"])
            if not progress:
                continue
            features = ready_features(
                int(row["input_tokens"]), int(row["prefix"]), progress["competitors"],
                progress["running"], progress["waiting"], progress["ready_competitors"],
                progress["batch_size"], progress["batch_tokens"],
            )
            if features is None:
                continue
            actual = float(row["P_next_other_ready_after_cut_ms"])
            series["Old 80K / 160 ms"].append([actual, consume_budget(progress["competitors"], 80_000., 160.)])
            series["Observed-progress estimate"].append([actual, max(0., predict(model["other_ready"], features))])
    return {"evidence": "DIAGNOSTIC_OFFLINE_SUPERVISION", "values": series}


def active_timeline(path):
    with path.open(newline="") as source:
        candidates = [row for row in csv.DictReader(source)
                      if row["cohort"].endswith("_5") and float(row["D_EC_export_RPC_ms"] or 0) > 0]
    if not candidates:
        return {"evidence": "NOT_MEASURED"}
    row = min(candidates, key=lambda entry: (entry["start"], entry["runtime_request_id"]))
    def bounds(left, right):
        return [[float(row[left]), float(row[right])]]

    events = {
        "Retired P cut → D submit": [[0, float(row["cut_to_D_submit_ms"])]],
        "D embedding load host interval": bounds("D_EC_begin_from_cut_ms", "D_EC_end_from_cut_ms"),
        "E descriptor export RPC": bounds("last_E_export_begin_from_cut_ms", "last_E_export_end_from_cut_ms"),
        "D submit → first output": bounds("cut_to_D_submit_ms", "D_cut_to_first_ms"),
    }
    return {
        "evidence": "MEASURED_ONLINE",
        "values": {"label": f"QPS5 D suffix of {row['suffix_tokens']} tokens",
                   "request_id": row["request_id"], "events": events},
    }


def online_series(inputs):
    from ..mllm_epd_h200_prefill_heavy_batch_matched.analysis import jsonl

    values = {}
    for item in inputs:
        label, separator, path = item.partition("=")
        if not separator or not label or label in values:
            raise ValueError("Online figure inputs use unique LABEL=PATH pairs")
        rows = jsonl(Path(path))
        values[label] = {
            "ttft_ms": [row["TTFT_ms"] for row in rows],
            "itl_ms": [gap for row in rows for gap in row["ITL_ms"]],
        }
    return {"evidence": "MEASURED_ONLINE", "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--aligned", nargs="+", type=Path)
    parser.add_argument("--timeline-stock", type=Path)
    parser.add_argument("--forecast-model", type=Path)
    parser.add_argument("--active-timing", type=Path)
    parser.add_argument("--online-series", action="append", default=[], help="LABEL=client request JSONL")
    parser.add_argument("--additional", type=Path, help="Other source-backed panels with the same schema")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    series = json.loads(args.additional.read_text()) if args.additional else {}
    if args.correctness:
        series["correctness"] = correctness(args.correctness)
    if args.aligned:
        series["funnel"] = funnel(args.aligned)
    if args.timeline_stock:
        series["timeline"] = stock_timeline(args.timeline_stock)
    if args.forecast_model:
        if args.timeline_stock is None:
            parser.error("Forecast inputs require the matching stock-cut CSV")
        series["predictor"] = forecast(args.timeline_stock, args.forecast_model)
    if args.active_timing:
        series["active_timeline"] = active_timeline(args.active_timing)
    if args.online_series:
        series["online"] = online_series(args.online_series)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(series, indent=2) + "\n")


if __name__ == "__main__":
    main()
