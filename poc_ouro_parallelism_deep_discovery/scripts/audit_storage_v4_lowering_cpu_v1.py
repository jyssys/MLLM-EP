"""Inspect rank-zero v4 host graphs and saved source. Never load binaries."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

from audit_storage_lowering_cpu_v1 import (
    Reader, artifact_summary, digest, graph_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def autotune_records(path):
    outer = Reader(path.read_bytes())
    parts = []
    while not outer.done():
        parts.append(outer.blob())
    bundle = Reader(parts[3])
    assert bundle.integer() == 1
    result = {}
    while not bundle.done():
        kind, count = bundle.string(), bundle.integer()
        for _ in range(count):
            key, payload = bundle.string(), bundle.blob()
            if kind == "autotune":
                result[key] = json.loads(payload)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = ROOT / "artifacts/storage_fully_isolated_checkpoint3.json"
    output = []
    for cell in json.loads(source.read_text())["records"]:
        arms = {}
        for arm in ["alias", "copies"]:
            base = Path(cell[arm + "_path"]) / "compile_cache/rank_0_0/backbone"
            arms[arm] = {"graph": graph_summary(base / "computation_graph.py"),
                         "pieces": artifact_summary(base / "vllm_compile_cache.py")}
        ga, gb = arms["alias"]["graph"], arms["copies"]["graph"]
        comparisons = {}
        common = sorted(set(arms["alias"]["pieces"]) & set(arms["copies"]["pieces"]), key=int)
        for key in common:
            a, b = arms["alias"]["pieces"][key], arms["copies"]["pieces"][key]
            aa, ab = autotune_records(Path(a["path"])), autotune_records(Path(b["path"]))
            tune_common = sorted(set(aa) & set(ab))
            stripped = lambda record: {k: v for k, v in record.items() if k != "time_taken_ms"}
            comparisons[key] = {
                "embedded_source_equal": a["embedded_source_sha256"] == b["embedded_source_sha256"],
                "normalized_call_ast_equal": a["call_ast_sha256"] == b["call_ast_sha256"],
                "triton_compute_hashes_equal": a["triton_compute_hashes"] == b["triton_compute_hashes"],
                "autotune_key_sets_equal": set(aa) == set(ab),
                "autotune_common_keys": len(tune_common),
                "autotune_values_equal": all(aa[k] == ab[k] for k in tune_common),
                "autotune_values_equal_without_time_taken_ms": all(stripped(aa[k]) == stripped(ab[k]) for k in tune_common),
                "autotune_non_timing_difference_keys": [k for k in tune_common if stripped(aa[k]) != stripped(ab[k])],
                "autotune_alias_only_keys": sorted(set(aa) - set(ab)),
                "autotune_copies_only_keys": sorted(set(ab) - set(aa)),
            }
        def compact_arm(value):
            graph = value["graph"]
            pieces = {}
            for key, piece in value["pieces"].items():
                calls = Counter(ast.unparse(ast.parse(call, mode="eval").body.func) for call in piece["runtime_calls"])
                pieces[key] = {field: piece[field] for field in [
                    "path", "sha256", "embedded_source_sha256", "call_ast_sha256", "triton_compute_hashes"]}
                pieces[key]["runtime_callee_counts"] = dict(calls)
            return {"graph": {field: graph[field] for field in [
                "path", "sha256", "leaf_count", "top_weight_arguments", "weight_depth_prefix_counts", "operations"]},
                    "pieces": pieces}
        output.append({
            "model": cell["model"], "P": cell["P"], "rank": 0,
            "host_operation_counts_equal": ga["operations"] == gb["operations"],
            "host_shape_annotations_equal": ga["shape_annotations"] == gb["shape_annotations"],
            "host_collective_attention_sequence_equal": ga["collective_attention_sequence_sha256"] == gb["collective_attention_sequence_sha256"],
            "different_normalized_leaf_indices": [i for i, (a, b) in enumerate(zip(ga["leaf_normalized_hashes"], gb["leaf_normalized_hashes"])) if a != b],
            "alias_only_piece_indices": sorted(set(arms["alias"]["pieces"]) - set(arms["copies"]["pieces"]), key=int),
            "copies_only_piece_indices": sorted(set(arms["copies"]["pieces"]) - set(arms["alias"]["pieces"]), key=int),
            "common_piece_comparisons": comparisons,
            "arms": {arm: compact_arm(value) for arm, value in arms.items()},
        })
    result = {"schema_version": 1, "source": str(source), "source_sha256": digest(source.read_bytes()),
                      "evidence_class": "CPU inspection of rank-zero saved graphs and generated source",
                      "limits": ["No compiled artifact was loaded or executed.",
                                 "A matching piece index is a source locator, not proof of equivalent runtime context.",
                                 "No numerical cause or timing cause is established."], "records": output}
    if args.output:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "cells": len(output)}))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
