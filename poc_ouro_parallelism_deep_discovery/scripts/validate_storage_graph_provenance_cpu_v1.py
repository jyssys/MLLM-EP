"""Check saved host graphs and export gzip copies. Do not import Torch."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re


BLOCK_PATTERN = re.compile(
    r"L_self_modules_campaign_execution_modules_(\d+)_modules_(\d+)_modules_"
    r"(.+)_parameters_weight_"
)
SHARED = {
    "L_self_modules_embed_tokens_parameters_weight_",
    "L_self_modules_norm_parameters_weight_",
}
ROLES = {
    "input_layernorm", "self_attn_modules_qkv_proj", "self_attn_modules_o_proj",
    "input_layernorm_2", "post_attention_layernorm",
    "mlp_modules_gate_up_proj", "mlp_modules_down_proj",
    "post_attention_layernorm_2",
}


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def expected_shapes(parallel):
    shapes = {role: "bf16[2048]" for role in ROLES}
    shapes.update({
        "self_attn_modules_qkv_proj": f"bf16[{6144 // parallel},2048]",
        "self_attn_modules_o_proj": f"bf16[2048,{2048 // parallel}]",
        "mlp_modules_gate_up_proj": f"bf16[{11264 // parallel},2048]",
        "mlp_modules_down_proj": f"bf16[2048,{5632 // parallel}]",
    })
    return shapes


def inspect_graph(payload, size, mode, parallel):
    """Check the current generated naming. Reject unknown naming."""
    layers = {"1.4B": 24, "2.6B": 48}[size]
    depths = [0] if mode == "alias" else list(range(4))
    expected_total = 8 * layers * len(depths) + 2
    expected_prefixes = {str(depth): 8 * layers for depth in depths}
    result = {
        "status": "FAIL", "expected_weight_arguments": expected_total,
        "expected_depth_prefix_counts": expected_prefixes, "failures": [],
        "source_sha256": sha256(payload), "source_bytes": len(payload),
    }
    try:
        tree = ast.parse(payload.decode("utf-8"))
        top = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        function = next(node for node in top.body
                        if isinstance(node, ast.FunctionDef) and node.name == "forward")
    except (UnicodeDecodeError, SyntaxError, StopIteration) as error:
        result["failures"].append(f"HOST_GRAPH_PARSE_ERROR:{type(error).__name__}")
        return result
    arguments = [arg for arg in function.args.args if "parameters_weight" in arg.arg]
    loads = Counter(node.id for node in ast.walk(function)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load))
    prefix_counts, slots, unknown, shared, bad_shapes = Counter(), [], [], [], []
    shape_contract = expected_shapes(parallel)
    weight_arguments = []
    for arg in arguments:
        annotation = (arg.annotation.value if isinstance(arg.annotation, ast.Constant)
                      and isinstance(arg.annotation.value, str) else None)
        weight_arguments.append({"name": arg.arg, "annotation": annotation,
                                 "source_load_count": loads[arg.arg]})
        normalized_shape = re.sub(r"\s+", "", annotation or "")
        match = BLOCK_PATTERN.fullmatch(arg.arg)
        if match:
            depth, layer, role = match.groups()
            prefix_counts[depth] += 1
            slots.append((int(depth), int(layer), role))
            expected_shape = shape_contract.get(role)
        elif arg.arg in SHARED:
            shared.append(arg.arg)
            expected_shape = (f"bf16[{49152 // parallel},2048]"
                              if "embed_tokens" in arg.arg else "bf16[2048]")
        else:
            unknown.append(arg.arg)
            expected_shape = None
        if expected_shape is None or normalized_shape != expected_shape:
            bad_shapes.append({"name": arg.arg, "actual": annotation,
                               "expected": expected_shape})
    expected_slots = {(depth, layer, role) for depth in depths
                      for layer in range(layers) for role in ROLES}
    unused = [arg.arg for arg in arguments if not loads[arg.arg]]
    result.update({
        "top_class": top.name, "top_weight_arguments": len(arguments),
        "depth_prefix_counts": dict(prefix_counts), "shared_weight_arguments": shared,
        "unknown_weight_arguments": unknown, "unused_weight_arguments": unused,
        "shape_failures": bad_shapes, "weight_arguments": weight_arguments,
        "missing_block_slots": sorted(expected_slots - set(slots)),
        "unexpected_block_slots": sorted(set(slots) - expected_slots),
    })
    checks = {
        "WEIGHT_ARGUMENT_COUNT": len(arguments) == expected_total,
        "STORAGE_DEPTH_PREFIX_COUNTS": dict(prefix_counts) == expected_prefixes,
        "SHARED_WEIGHT_NAMES": set(shared) == SHARED and len(shared) == 2,
        "BLOCK_ROLE_AND_LAYER_COVERAGE": set(slots) == expected_slots,
        "DUPLICATE_BLOCK_SLOT": len(slots) == len(set(slots)),
        "UNRECOGNIZED_WEIGHT_NAME": not unknown,
        "UNUSED_WEIGHT_ARGUMENT": not unused,
        "WEIGHT_SHAPE": not bad_shapes,
    }
    result["failures"].extend(code for code, valid in checks.items() if not valid)
    result["status"] = "PASS" if not result["failures"] else "FAIL"
    return result


def export_job(cache_dir, size, mode, parallel, output_dir):
    """Create a new evidence directory. Never overwrite existing evidence."""
    cache_dir = Path(cache_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = sorted(cache_dir.glob("rank_*_*/backbone/computation_graph.py"))
    expected_ranks = {f"rank_{rank}_0" for rank in range(parallel)}
    actual_ranks = {path.parent.parent.name for path in paths}
    records = []
    for path in paths:
        payload = path.read_bytes()
        record = inspect_graph(payload, size, mode, parallel)
        rank = path.parent.parent.name
        compressed = gzip.compress(payload, mtime=0)
        export = output_dir / f"{rank}_computation_graph.py.gz"
        export.write_bytes(compressed)
        record.update({"source_path": str(path), "rank": rank,
                       "gzip_path": str(export), "gzip_sha256": sha256(compressed),
                       "gzip_bytes": len(compressed)})
        records.append(record)
    failures = []
    if actual_ranks != expected_ranks or len(paths) != parallel:
        failures.append("RANK_SET_MISMATCH")
    if any(record["status"] != "PASS" for record in records):
        failures.append("HOST_GRAPH_PROVENANCE_MISMATCH")
    manifest = {
        "schema_version": 1, "validator": Path(__file__).name,
        "validator_sha256": sha256(Path(__file__).read_bytes()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if failures else "PASS", "failures": failures,
        "size": size, "mode": mode, "P": parallel, "cache_dir": str(cache_dir),
        "expected_ranks": sorted(expected_ranks), "actual_ranks": sorted(actual_ranks),
        "scope": "Saved host-graph provenance only.",
        "limits": [
            "This check does not prove runtime parameter addresses or numerical parity.",
            "This check does not validate AOT cache isolation or timing.",
            "The exporter does not load compiled artifacts or export binary caches.",
        ],
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--size", required=True, choices=["1.4B", "2.6B"])
    parser.add_argument("--mode", required=True, choices=["alias", "copies"])
    parser.add_argument("--P", required=True, type=int, choices=[1, 2, 4])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = export_job(args.cache_dir, args.size, args.mode, args.P, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ["status", "failures", "actual_ranks"]}))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
