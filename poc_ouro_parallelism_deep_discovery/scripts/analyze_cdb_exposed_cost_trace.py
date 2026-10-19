"""Validate retained outer observations. Invalid timing never receives E2E attribution."""
import argparse
import collections
import datetime as dt
import statistics
from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha, write
from poc_ouro_parallelism_deep_discovery.cdb_cost_observer import validate_triplet
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import table, svg


def analyze(job, output):
    output.mkdir(parents=True, exist_ok=False)
    guard = read(job / "guard_status.json")
    assert guard["success"] and guard["exit_code"] == 0
    root = job / "cost_trace"
    status = read(root / "status.json")
    plan = read(root / "execution_plan.json")
    assert status["success"]
    requests, intervals, provenance = [], [], {}
    for number, request_id in enumerate(plan["request_ids"], 1):
        path = root / request_id.replace("/", "_")
        phases = {name: read(path / (name.lower() + ".json")) for name in (
            "UNTIMED_SAME_REQUEST_WARM", "CLEAN_BEFORE", "TRACE", "CLEAN_AFTER")}
        result = validate_triplet(phases)
        assert result == read(path / "validity.json")
        assert not result["errors"]
        trace = read(path / "trace_intervals.json")
        export = read(path / "trace_export_status.json")
        assert export["elapsed_times_resolved"] and export["observer_overrides_restored"]
        assert [row["id"] for row in trace] == list(range(len(trace)))
        observed = collections.Counter(row["event_kind"] for row in trace if "event_kind" in row)
        assert dict(observed) == {k: v for k, v in phases["TRACE"]["event_counts"].items() if v}
        by_id = {row["id"]: row for row in trace}
        for row in trace:
            assert row["host_start_ns"] <= row["host_end_ns"]
            assert phases["TRACE"]["host_start_ns"] <= row["host_start_ns"]
            assert row["host_end_ns"] <= phases["TRACE"]["host_end_ns"]
            if row["parent_id"] is not None:
                parent = by_id[row["parent_id"]]
                assert parent["host_start_ns"] <= row["host_start_ns"] <= row["host_end_ns"] <= parent["host_end_ns"]
            if "device_span_ms" in row:
                assert row["device_span_ms"] >= 0
        groups = collections.defaultdict(list)
        for row in trace:
            groups[row["kind"]].append(row)
        for kind, rows in sorted(groups.items()):
            device = [row["device_span_ms"] for row in rows if "device_span_ms" in row]
            intervals.append(dict(request_id=request_id, kind=kind, calls=len(rows),
                device_pairs=len(device), host_span_sum_ms=sum(row["host_span_ms"] for row in rows),
                device_span_sum_ms=sum(device) if device else None,
                device_span_median_ms=statistics.median(device) if device else None,
                validity=result["validity"], additive_E2E_attribution=False,
                interpretation="Nested observed windows. Invalid observer guard. Not clean cost or removable work."))
        depths = phases["TRACE"]["depth_events"]
        requests.append(dict(request_number=number, request_id=request_id,
            output_tokens=phases["TRACE"]["generated_tokens"],
            finish_reason=phases["TRACE"]["finish_reason"], decode_tokens=len(depths),
            decode_passes=phases["TRACE"]["stats"]["recurrent_batches"],
            publication_windows=observed["publication"],
            clean_before_ms=phases["CLEAN_BEFORE"]["wall_ms"],
            trace_ms=phases["TRACE"]["wall_ms"], clean_after_ms=phases["CLEAN_AFTER"]["wall_ms"],
            clean_drift_percent=100 * result["clean_drift_fraction"],
            trace_over_clean_mean_percent=100 * result["trace_over_clean_mean_minus_one"],
            output_depth_NFE_parity=True, measured_graph_captures=0,
            validity=result["validity"], quantitative_bottleneck_attribution=result["attribution_permitted"]))
        for source in sorted(path.glob("*.json")):
            provenance[str(source.relative_to(ROOT))] = sha(source)
    table(output / "request_validity.csv", requests)
    table(output / "nested_intervals_diagnostic_only.csv", intervals)
    svg(output / "observer_wall_triplets.svg", [dict(name=name, points=[
        (row["request_number"], row[field]) for row in requests]) for name, field in (
        ("Clean before", "clean_before_ms"), ("Trace: invalid observer", "trace_ms"),
        ("Clean after", "clean_after_ms"))], "CDB outer observer: validity diagnosis only",
        "Frozen calibration request number", "Request wall (ms)")
    summary = dict(status="OBSERVATION_VALIDITY_FAILED", size=plan["size"], mode=plan["mode"],
        requests=requests, accepted_attribution_requests=sum(r["quantitative_bottleneck_attribution"] for r in requests),
        other_cells="NOT_RUN. Suspend this diagnostic lane after its frozen observer/drift guards fail.",
        candidate_speedup="NOT_RUN", quality="NOT_EVALUATED. Cap128 is a hardware diagnosis, not benchmark quality.",
        confidence_interval="NOT_ESTIMATED. One triplet per request.",
        claims_forbidden=["Additive E2E component shares", "Amdahl headroom", "Clean publication/gate cost",
                          "Pure CPU remainder", "Mode-to-mode or size-to-size timing conclusion"],
        source_sha256=provenance, implementation_sha256=plan["implementation_sha256"],
        analyzed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(ROOT / args.job, ROOT / args.output)
    print(result["status"], len(result["requests"]), result["accepted_attribution_requests"])


if __name__ == "__main__":
    main()
