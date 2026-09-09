"""Create rank-critical tables for the CP x EP direct-handoff PoC."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path


def load_rows(pattern: str, key: str) -> tuple[list[dict], list[dict]]:
    documents = [json.loads(Path(path).read_text()) for path in glob.glob(pattern)]
    return [row for document in documents for row in document[key]], documents


def median_critical(rows: list[dict], filters: dict, key: str) -> float:
    selected = [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    repeats = sorted({row["repeat"] for row in selected})
    values = [max(row[key] for row in selected if row["repeat"] == repeat) for repeat in repeats]
    return float(statistics.median(values))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    crossover_rows = []
    for directory in ("cp_crossover_layer24", "cp_crossover_64k"):
        rows, _ = load_rows(str(root / directory / "cp_attention_rank*.json"), "rows")
        for length in sorted({row["length"] for row in rows}):
            for cp in sorted({row["cp"] for row in rows}):
                item = {"layer": 24, "length": length, "cp": cp, "repetitions": 30}
                for key in (
                    "qkv_norm_ms", "qkv_a2a_ms", "attention_ms",
                    "return_a2a_ms", "o_proj_ms", "attention_path_ms",
                ):
                    item[key] = median_critical(rows, {"length": length, "cp": cp}, key)
                crossover_rows.append(item)
    write_csv(root / "CP_CROSSOVER.csv", crossover_rows)

    boundary_rows = []
    boundary_directories = {
        4: "boundary_layer4",
        24: "boundary_layer24",
        47: "boundary_layer47",
    }
    for layer, directory in boundary_directories.items():
        rows, _ = load_rows(str(root / directory / "cp_ep_boundary_rank*.json"), "rows")
        for length in sorted({row["length"] for row in rows}):
            item = {"layer": layer, "length": length, "repetitions": 30}
            for key in (
                "qkv_norm_ms", "qkv_a2a_ms", "attention_ms", "return_a2a_ms",
                "o_proj_ms", "residual_norm_ms", "router_topk_ms",
                "dispatch_critical_ms", "expert_ms", "combine_critical_ms",
                "attention_path_ms", "cp_to_first_expert_ms", "moe_path_ms",
                "layer_path_ms",
            ):
                item[key] = median_critical(rows, {"length": length}, key)
            item["perfect_block_overlap_oracle_pct"] = (
                item["layer_path_ms"]
                - max(item["attention_path_ms"], item["moe_path_ms"])
            ) / item["layer_path_ms"] * 100
            item["naive_return_plus_dispatch_pct"] = (
                item["return_a2a_ms"] + item["dispatch_critical_ms"]
            ) / item["layer_path_ms"] * 100
            boundary_rows.append(item)
    rows, _ = load_rows(str(root / "boundary_64k_l24" / "cp_ep_boundary_rank*.json"), "rows")
    item = {"layer": 24, "length": 65536, "repetitions": 30}
    for key in (
        "qkv_norm_ms", "qkv_a2a_ms", "attention_ms", "return_a2a_ms",
        "o_proj_ms", "residual_norm_ms", "router_topk_ms",
        "dispatch_critical_ms", "expert_ms", "combine_critical_ms",
        "attention_path_ms", "cp_to_first_expert_ms", "moe_path_ms", "layer_path_ms",
    ):
        item[key] = median_critical(rows, {"length": 65536}, key)
    item["perfect_block_overlap_oracle_pct"] = (
        item["layer_path_ms"] - max(item["attention_path_ms"], item["moe_path_ms"])
    ) / item["layer_path_ms"] * 100
    item["naive_return_plus_dispatch_pct"] = (
        item["return_a2a_ms"] + item["dispatch_critical_ms"]
    ) / item["layer_path_ms"] * 100
    boundary_rows.append(item)
    write_csv(root / "HANDOFF_BREAKDOWN.csv", boundary_rows)

    streaming_rows = []
    streaming_directories = {
        4: "streaming_layer4",
        24: "streaming_formal_l24",
        47: "streaming_layer47",
    }
    for layer, directory in streaming_directories.items():
        rows, documents = load_rows(str(root / directory / "streaming_rank*.json"), "rows")
        correctness = [row for document in documents for row in document["correctness"]]
        for length in sorted({row["length"] for row in rows}):
            serial = median_critical(rows, {"length": length, "variant": "serial"}, "wall_ms")
            for block in sorted({row["block"] for row in rows if row["length"] == length and row["block"]}):
                no_overlap = median_critical(rows, {"length": length, "variant": "chunk_no_overlap", "block": block}, "wall_ms")
                streaming = median_critical(rows, {"length": length, "variant": "streaming", "block": block}, "wall_ms")
                checks = [row for row in correctness if row["length"] == length and row["block"] == block]
                streaming_rows.append({
                    "layer": layer,
                    "length": length,
                    "block": block,
                    "repetitions": 30,
                    "serial_ms": serial,
                    "same_chunk_no_overlap_ms": no_overlap,
                    "streaming_ms": streaming,
                    "streaming_gain_vs_same_chunk_pct": (no_overlap - streaming) / no_overlap * 100,
                    "streaming_gain_vs_serial_pct": (serial - streaming) / serial * 100,
                    "min_cosine": min(row["cosine"] for row in checks),
                    "max_relative_l2": max(row["relative_l2"] for row in checks),
                    "min_route_agreement": min(row["route_agreement"] for row in checks),
                })
    rows, documents = load_rows(str(root / "streaming_64k_l24" / "streaming_rank*.json"), "rows")
    correctness = [row for document in documents for row in document["correctness"]]
    serial = median_critical(rows, {"length": 65536, "variant": "serial"}, "wall_ms")
    no_overlap = median_critical(rows, {"length": 65536, "variant": "chunk_no_overlap", "block": 1024}, "wall_ms")
    streaming = median_critical(rows, {"length": 65536, "variant": "streaming", "block": 1024}, "wall_ms")
    streaming_rows.append({
        "layer": 24, "length": 65536, "block": 1024, "repetitions": 30,
        "serial_ms": serial, "same_chunk_no_overlap_ms": no_overlap,
        "streaming_ms": streaming,
        "streaming_gain_vs_same_chunk_pct": (no_overlap - streaming) / no_overlap * 100,
        "streaming_gain_vs_serial_pct": (serial - streaming) / serial * 100,
        "min_cosine": min(row["cosine"] for row in correctness),
        "max_relative_l2": max(row["relative_l2"] for row in correctness),
        "min_route_agreement": min(row["route_agreement"] for row in correctness),
    })
    write_csv(root / "STREAMING_ORACLE.csv", streaming_rows)

    clean_rows, clean_documents = load_rows(
        str(root / "streaming_clean_l24" / "streaming_rank*.json"), "rows"
    )
    clean_correctness = [
        row for document in clean_documents for row in document["correctness"]
    ]
    clean_summary = []
    for length in sorted({row["length"] for row in clean_rows}):
        serial = median_critical(
            clean_rows, {"length": length, "variant": "serial"}, "wall_ms"
        )
        for block in sorted(
            {row["block"] for row in clean_rows if row["length"] == length and row["block"]}
        ):
            no_overlap = median_critical(
                clean_rows,
                {"length": length, "variant": "chunk_no_overlap", "block": block},
                "wall_ms",
            )
            streaming = median_critical(
                clean_rows,
                {"length": length, "variant": "streaming", "block": block},
                "wall_ms",
            )
            checks = [
                row
                for row in clean_correctness
                if row["length"] == length and row["block"] == block
            ]
            detailed = next(
                (
                    row
                    for row in streaming_rows
                    if row["layer"] == 24
                    and row["length"] == length
                    and row["block"] == block
                ),
                None,
            )
            clean_summary.append(
                {
                    "layer": 24,
                    "length": length,
                    "block": block,
                    "repetitions": 30,
                    "serial_ms": serial,
                    "same_chunk_no_overlap_ms": no_overlap,
                    "streaming_ms": streaming,
                    "streaming_gain_vs_same_chunk_pct": (no_overlap - streaming)
                    / no_overlap
                    * 100,
                    "streaming_gain_vs_serial_pct": (serial - streaming) / serial * 100,
                    "detailed_streaming_ms": (
                        detailed["streaming_ms"] if detailed is not None else ""
                    ),
                    "observer_tax_pct_of_clean_streaming": (
                        (detailed["streaming_ms"] - streaming) / streaming * 100
                        if detailed is not None
                        else ""
                    ),
                    "min_cosine": min(row["cosine"] for row in checks),
                    "max_relative_l2": max(row["relative_l2"] for row in checks),
                    "min_route_agreement": min(row["route_agreement"] for row in checks),
                }
            )
    write_csv(root / "STREAMING_CLEAN_TIMING.csv", clean_summary)

    # Byte lower bound for exact Ulysses -> routed EP transport.  For each
    # token, Ulysses must communicate P-1 of P attention head shards.  After
    # exact routing, DeepEP must still transport one full-H activation to each
    # remote destination rank.  Route-aware transport cannot choose those
    # destinations before exact residual/norm/router evaluation.
    route_docs = [
        json.loads(Path(path).read_text())
        for path in glob.glob(str(root / "direct_transport_route_stats" / "cp_ep_boundary_rank*.json"))
    ]
    direct_rows = []
    for length in (8192, 16384, 32768):
        mean_fanout = statistics.mean(doc["route_summaries"][str(length)]["mean_rank_fanout"] for doc in route_docs)
        mean_remote = statistics.mean(doc["route_summaries"][str(length)]["mean_remote_destinations"] for doc in route_docs)
        boundary = next(row for row in boundary_rows if row["layer"] == 24 and row["length"] == length)
        cp_bytes = 2048 * 2 * 3 / 4
        ep_bytes = mean_remote * 2048 * 2
        direct_rows.append({
            "layer": 24,
            "length": length,
            "mean_fanout": mean_fanout,
            "mean_remote_destinations": mean_remote,
            "ulysses_return_network_bytes_per_token": cp_bytes,
            "ep_dispatch_network_bytes_per_token": ep_bytes,
            "baseline_network_bytes_per_token": cp_bytes + ep_bytes,
            "exact_direct_network_lower_bound_bytes_per_token": cp_bytes + ep_bytes,
            "exact_bytes_saved_pct": 0.0,
            "invalid_naive_remove_return_and_dispatch_time_pct": boundary["naive_return_plus_dispatch_pct"],
            "valid_exact_direct_layer_oracle_pct": 0.0,
            "decision": "ALGEBRAICALLY_BLOCKED_ROUTE_DEPENDENCY",
        })
    write_csv(root / "DIRECT_TRANSPORT_ORACLE.csv", direct_rows)


if __name__ == "__main__":
    main()
