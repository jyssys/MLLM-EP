"""Recreate portable correctness CSVs from original analysis receipts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .clone_receipt import clonelist, jsonl
from .native_aa import compare_request_files


def load(labeled_path):
    label, file_name = labeled_path.split("=", 1)
    path = Path(file_name)
    data = path.read_bytes()
    return label, hashlib.sha256(data).hexdigest(), json.loads(data)


def geometry_row(regime, sha256, packet, label=None):
    rows = packet.get("cases", [])
    split = [row for row in rows if row["actual_offloaded_tokens"] > 0]
    receipt = {
        "regime": regime,
        "transport": "slice" if regime.endswith("_slice") else "whole",
        "fixture": label or packet["case"],
        "evidence": packet["evidence"],
        "N8_evidence": packet["evidence"],
        "N1_passed": None,
        "N1_greedy_exact": None,
        "N1_D_final_KV_byte_equal_layers": None,
        "N1_teacher_output_tokens": None,
        "N1_candidate_output_tokens": None,
        "N8_passed": packet["passed"],
        "requests": len(rows),
        "exact_greedy": sum(row["late_vs_native_exact_greedy"] for row in rows),
        "identical_embedding": packet.get("verified_same_E"),
        "prefix_KV_byte_equal_layers": sum(
            row["numerical_checks"]["late_D_vs_native_D"]["prefix"]["strict_equal"] for row in rows
        ),
        "suffix_KV_byte_equal_layers": sum(
            row["numerical_checks"]["late_D_vs_native_D"]["suffix"]["strict_equal"] for row in rows
        ),
        "worst_suffix_KV_relative_L2": max(
            (row["numerical_checks"]["late_D_vs_native_D"]["suffix"]["max_relative_L2"] for row in rows), default=None
        ),
        "actual_splits": len(split),
        "actual_offload_tokens": sum(row["actual_offloaded_tokens"] for row in split),
        "nominal_ratio": packet.get("ratio"),
        "realized_ratio": split[0]["actual_offloaded_tokens"]/split[0]["input_length"] if len(split) == 1 else None,
        "shared_decode_steps": (packet.get("lead_shared_decode") or {}).get("late", {}).get("shared_steps"),
        "shared_decode_max": (packet.get("lead_shared_decode") or {}).get("late", {}).get("max_decode_batch"),
        "N8_D_cloned_bytes": sum(
            row["logical_bytes"] for row in packet.get("D_transport", {}).get("clones", [])
        ) if "D_transport" in packet else None,
        "N8_D_smaller_clone_count": packet.get("D_transport", {}).get("strictly_smaller_clones"),
        "D_transport_verified": None,
        "fixture_passed": packet["passed"],
        "analysis_SHA256": sha256,
    }
    return receipt


def staged_rows(geometry, ratio):
    output = []
    for raw in geometry:
        label, sha256, receipt = load(raw)
        output.extend(geometry_row(label, sha256, case) for case in receipt["cases"])
    for raw in ratio:
        label, sha256, receipt = load(raw)
        for case in receipt["cases"]:
            row = geometry_row(label, sha256, case["N8"], label=case["name"])
            exact_n1_layers = case["N1"].get("checks", {}).get("exact_d_final_prompt_layers")
            transport_checks = [case[stage].get("D_transport") for stage in ("N1", "N8")]
            stages = {case[stage].get("evidence") for stage in ("N1", "N8")}
            evidence = "CORRECTNESS_PASS" if case["passed"] else (
                "FAILED_EXECUTION" if "FAILED_EXECUTION" in stages else "DIAGNOSTIC"
            )
            row.update(evidence=evidence, N1_passed=case["N1"]["passed"],
                       N1_greedy_exact=case["N1"].get("exact_greedy_tokens"),
                       N1_D_final_KV_byte_equal_layers=sum(exact_n1_layers) if exact_n1_layers is not None else None,
                       N1_teacher_output_tokens=case["N1"].get("teacher_output_tokens"),
                       N1_candidate_output_tokens=case["N1"].get("observed_output_tokens"),
                       nominal_ratio=case["nominal_ratio"],
                       realized_ratio=case["actual_ratio"] if row["actual_splits"] else None,
                       D_transport_verified=all(check["verified_range"] for check in transport_checks)
                       if all(check is not None for check in transport_checks) else None,
                       fixture_passed=case["passed"])
            output.append(row)
    return output


def recovery_rows(inputs):
    comparisons, logits, layers = [], [], []
    for raw in inputs:
        label, sha256, receipt = load(raw)
        for case in receipt["cases"]:
            for contrast in ("serial_AA", "mixed_AA", "late_vs_native"):
                for row in case[contrast]:
                    comparisons.append({
                        "regime": label, "fixture": case["name"], "contrast": contrast,
                        "id": row["id"], "greedy_exact": row["greedy_exact"], "E_byte_equal": row["E_byte_equal"],
                        "P_geometry_equal": row["P_geometry_equal"], "D_geometry_equal": row["D_geometry_equal"],
                        "first_distinct_token": row["first_distinct_token"], "analysis_SHA256": sha256,
                    })
            for step in case["teacher"]["steps"]:
                logits.append({
                    "regime": label, "fixture": case["name"], "id": case["request_id"], "step": step["step"],
                    "greedy_differs": step["greedy_differs"], "native_greedy": step["native"]["greedy_id"],
                    "late_greedy": step["candidate"]["greedy_id"], "matched_input": step["native"]["input_token_id"],
                    "native_margin": step["native"]["top1_top2_margin"],
                    "late_margin": step["candidate"]["top1_top2_margin"],
                    "native_top1_id": step["native"]["token_ids"][0], "native_top2_id": step["native"]["token_ids"][1],
                    "native_top1_logit": step["native"]["logits"][0], "native_top2_logit": step["native"]["logits"][1],
                    "late_top1_id": step["candidate"]["token_ids"][0], "late_top2_id": step["candidate"]["token_ids"][1],
                    "late_top1_logit": step["candidate"]["logits"][0], "late_top2_logit": step["candidate"]["logits"][1],
                    "native_prepared": step["geometry"][0]["actual_batch"],
                    "late_prepared": step["geometry"][1]["actual_batch"],
                    "native_padded_rows": step["geometry"][0]["padded_query_rows"],
                    "late_padded_rows": step["geometry"][1]["padded_query_rows"],
                    "logits_relative_L2": step["tensor_differences"]["logits"]["relative_L2"],
                    "hidden_relative_L2": step["tensor_differences"]["final_hidden"]["relative_L2"],
                    "analysis_SHA256": sha256,
                })
                for section, differences in step["tensor_differences"].items():
                    if not section.startswith("layer"):
                        continue
                    layer, tensor = section.split(".", 1)
                    layers.append({
                        "regime": label, "fixture": case["name"], "step": step["step"],
                        "layer": int(layer[5:]), "tensor": tensor, **differences, "analysis_SHA256": sha256,
                    })
    return comparisons, logits, layers


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def native_rows(inputs):
    rows = []
    for receipt in inputs:
        label, files = receipt.split("=", 1)
        a, b = (Path(name) for name in files.split(",", 1))
        digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b)]
        for result in compare_request_files(a, b):
            rows.append({
                "regime": label, "request_id": result["request_id"], "exact": result["exact"],
                "first_output_tokens": result["output_tokens"][0], "second_output_tokens": result["output_tokens"][1],
                "first_distinct_token": result["first_distinct_token"],
                "first_raw_SHA256": digests[0], "second_raw_SHA256": digests[1],
            })
    return rows


def transport_state_rows(inputs):
    rows = []
    for receipt in inputs:
        fixture, files = receipt.split("=", 1)
        a, b = (Path(name) for name in files.split(",", 1))
        manifests = [json.loads(path.read_text()) for path in (a, b)]
        contracts = [manifest["contracts"] for manifest in manifests]
        receipts = [manifest["full_prompt_tensor_files"] for manifest in manifests]
        for index, (left, right) in enumerate(zip(*map(sorted, receipts), strict=True)):
            if contracts[0][left] != contracts[1][right]:
                raise ValueError("Same-cut transport state fixtures changed prompt geometry")
            hashes = [receipt[key]["SHA256"] for receipt, key in zip(receipts, (left, right), strict=True)]
            rows.append({
                "fixture": fixture, "request_index": index, "whole_full_prompt_SHA256": hashes[0],
                "slice_full_prompt_SHA256": hashes[1], "full_state_byte_equal": hashes[0] == hashes[1],
            })
    return rows


def ratio_transport_rows(inputs):
    rows = []
    for receipt in inputs:
        regime, file_name = receipt.split("=", 1)
        if not regime.endswith("_slice"):
            continue
        path = Path(file_name)
        raw = path.parent.parent / "raw"
        trace = {row["runtime_request_id"]: row for row in jsonl(raw / "proxy_epd.jsonl")}
        for case in json.loads(path.read_text())["cases"]:
            for stage, suffix in (("N1", "_n1_0000"), ("N8", "_n8_late_0000")):
                rid = f"ratio_{case['name']}{suffix}"
                record = trace.get(rid, {})
                plan = record.get("late_plan")
                if not plan or not any(row["accepted"] for row in plan["decisions"]):
                    continue
                checks = clonelist(raw, plan["image_hashes"], plan["image_spans"], case["cut"])
                for clone in checks["clones"]:
                    rows.append({
                        "regime": regime, "fixture": case["name"], "stage": stage, "fixture_passed": case["passed"],
                        "verified_range": checks["verified_range"], "source_rows": clone["source_shape"][0],
                        "first_source_row": clone["first_source_row"], "clone_rows": clone["clone_shape"][0],
                        "logical_bytes": clone["logical_bytes"], "copy_before_ACK": clone["copy_before_ACK"],
                    })
    return rows


def continuation_rows(inputs):
    rows = []
    for raw in inputs:
        label, sha256, receipt = load(raw)
        result = receipt.get("natural_hold_continuation")
        if not result:
            continue
        checks, offer = result.get("checks", {}), result.get("offer", {})
        control = receipt.get("native_epd_resume_control", {})
        paired = control.get("actual_geometry_and_state", {})
        counts = control.get("output_counts", (None, None))
        rows.append({
            "regime": label, "passed": result["passed"], "evidence": result["evidence"],
            "request_id": result.get("request_id"), "prefix_tokens": result.get("prefix_tokens"),
            "full_prompt_tokens": result.get("full_prompt_tokens"), "greedy_exact": result.get("exact_greedy_tokens"),
            "teacher_output_tokens": result.get("teacher_output_tokens"),
            "candidate_output_tokens": result.get("observed_output_tokens"),
            "P_final_KV_exact_layers": sum(checks.get("exact_p_final_prompt_layers", [])),
            "D_prefix_KV_exact_layers": sum(checks.get("exact_imported_prefix_layers", [])),
            "same_request_no_repeat": checks.get("no_repeated_query_rows_same_engine_request"),
            "prefill_recomputations": offer.get("prefill_recomputations"),
            "reason": offer.get("reason", result.get("error")),
            "native_EPD_paired_evidence": control.get("evidence"),
            "native_EPD_paired_greedy_exact": control.get("Paired_Greedy_exact"),
            "native_EPD_output_tokens": counts[0], "native_EPD_resume_output_tokens": counts[1],
            "native_EPD_paired_E_exact": control.get("E_embedding_exact"),
            "native_EPD_paired_P_geometry": paired.get("P", {}).get("same_geometry"),
            "native_EPD_paired_D_geometry": paired.get("D", {}).get("same_geometry"),
            "native_EPD_paired_P_KV_layers": paired.get("P", {}).get("prompt_KV_exact_layers"),
            "native_EPD_paired_D_KV_layers": paired.get("D", {}).get("prompt_KV_exact_layers"),
            "analysis_SHA256": sha256,
        })
    return rows


def eager_snapshot_rows(inputs):
    rows = []
    for item in inputs:
        regime, file_name = item.split("=", 1)
        if not regime.startswith("eager"):
            continue
        path = Path(file_name)
        captures = path.parent.parent / "raw/batch_prefix"
        for fixture in json.loads(path.read_text())["cases"]:
            for role in ("P", "D"):
                manifest_path = captures / f"ratio_{fixture['name']}_n1_{role}" / "manifest.json"
                if not manifest_path.is_file():
                    continue
                _, sha256, manifest = load(f"{regime}={manifest_path}")
                for event in manifest["events"]:
                    for record in event["requests"]:
                        if record["phase"] == "prefill" and "inputs_embeds" in record:
                            rows.append({
                                "regime": regime, "fixture": fixture["name"], "stage": role,
                                "query_start": record["start"], "query_rows": record["query_tokens"],
                                "pre_forward_embedding_snapshot": manifest.get("pre_forward_embedding_snapshot", False),
                                "changed_during_forward": record.get("inputs_embeds_changed_during_forward"),
                                "N1_exact_gate_passed": fixture["N1"]["passed"], "manifest_SHA256": sha256,
                            })
    return rows


def native_cost_rows(inputs):
    rows = []
    for raw in inputs:
        label, sha256, receipt = load(raw)
        if not receipt["primary_execution_valid"]:
            raise ValueError("Native cost row requires a complete cohort")
        rows.append({
            "regime": label, "cohort": receipt["run"], "completed": receipt["completed_count"],
            "TTFT_p95_ms": receipt["TTFT_p95_ms"], "ITL_p99_ms": receipt["ITL_p99_ms"],
            "TPOT_p95_ms": receipt["TPOT_p95_ms"], "ITL_PASS": receipt["gate_attainment"]["ITL_PASS"],
            "TPOT_PASS": receipt["gate_attainment"]["TPOT_PASS"], "SLO_goodput": receipt["SLO_goodput"],
            "analysis_SHA256": sha256,
        })
    return rows


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--out", required=True, type=Path)
    args.add_argument("--geometry", action="append", default=[], metavar="REGIME=FILE")
    args.add_argument("--ratio", action="append", default=[], metavar="REGIME=FILE")
    args.add_argument("--recovery", action="append", default=[], metavar="REGIME=FILE")
    args.add_argument("--native-pair", action="append", default=[], metavar="REGIME=FIRST,SECOND")
    args.add_argument("--same-cut-state", action="append", default=[], metavar="FIXTURE=WHOLE,SLICE")
    args.add_argument("--native-cost", action="append", default=[], metavar="REGIME=FILE")
    options = args.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    write_csv(options.out / "strict_geometry.csv", staged_rows(options.geometry, options.ratio))
    comparisons, logits, layers = recovery_rows(options.recovery)
    write_csv(options.out / "native_aa_geometry.csv", comparisons)
    write_csv(options.out / "teacher_divergence.csv", logits)
    write_csv(options.out / "teacher_layer_divergence.csv", layers)
    write_csv(options.out / "native_aa_requests.csv", native_rows(options.native_pair))
    write_csv(options.out / "same_cut_full_prompt_state.csv", transport_state_rows(options.same_cut_state))
    write_csv(options.out / "ratio_D_clone_rows.csv", ratio_transport_rows(options.ratio))
    write_csv(options.out / "natural_hold_continuation.csv", continuation_rows(options.ratio))
    write_csv(options.out / "eager_embedding_probe.csv", eager_snapshot_rows(options.ratio))
    write_csv(options.out / "native_numeric_cost.csv", native_cost_rows(options.native_cost))


if __name__ == "__main__":
    main()
