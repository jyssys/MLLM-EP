"""Read job-local graphs and embedded source. Never load compiled artifacts."""

import ast
import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import pickletools
import re
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


class Reader:
    def __init__(self, payload):
        assert zlib.crc32(payload[4:]).to_bytes(4, "big") == payload[:4]
        self.payload, self.position = payload, 4

    def integer(self):
        end = self.position + 8
        result = int.from_bytes(self.payload[self.position:end], "big")
        self.position = end
        return result

    def blob(self):
        end = self.position + 8
        size = int.from_bytes(self.payload[self.position:end], "big")
        start = end
        self.position = end + size
        assert self.position <= len(self.payload)
        return self.payload[start:self.position]

    def string(self):
        return base64.b64decode(self.blob()).decode()

    def done(self):
        return self.position == len(self.payload)


def embedded_source(path):
    outer = Reader(path.read_bytes())
    parts = []
    while not outer.done():
        parts.append(outer.blob())
    assert len(parts) == 4 and parts[0] == b"CacheCompiledArtifact"
    bundle = Reader(parts[3])
    assert bundle.integer() == 1
    sources = []
    while not bundle.done():
        kind, count = bundle.string(), bundle.integer()
        for _ in range(count):
            key, payload = bundle.string(), bundle.blob()
            if kind != "inductor":
                continue
            # Inspect pickle opcodes. Do not unpickle or import saved objects.
            for opcode, argument, offset in pickletools.genops(payload):
                if isinstance(argument, str) and "def call(" in argument:
                    sources.append((key, argument))
    assert len(sources) == 1
    return sources[0]


def bundle_entry_hashes(path):
    """Read cache-entry hashes. Do not deserialize saved objects."""
    outer = Reader(path.read_bytes())
    parts = []
    while not outer.done():
        parts.append(outer.blob())
    assert len(parts) == 4 and parts[0] == b"CacheCompiledArtifact"
    bundle = Reader(parts[3])
    assert bundle.integer() == 1
    result = {}
    while not bundle.done():
        kind, count = bundle.string(), bundle.integer()
        entries = []
        for _ in range(count):
            key, payload = bundle.string(), bundle.blob()
            entries.append((key, digest(payload)))
        result[kind] = sorted(entries)
    return result


class Canonical(ast.NodeTransformer):
    def __init__(self, function):
        self.names = {arg.arg: f"arg{i}" for i, arg in enumerate(function.args.args)}

    def visit_arg(self, node):
        node.arg = self.names[node.arg]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in {"torch", "slice", "getattr", "len", "int", "float", "bool"}:
            return node
        if node.id not in self.names:
            self.names[node.id] = f"local{len(self.names)}"
        node.id = self.names[node.id]
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = re.sub(r"\bs\d+\b", "S", node.value)
        return node


def normalized_function(function):
    function = copy.deepcopy(function)
    function.decorator_list = []
    return ast.dump(Canonical(function).visit(function), include_attributes=False)


def graph_summary(path):
    tree = ast.parse(path.read_text())
    classes = [item for item in ast.walk(tree) if isinstance(item, ast.ClassDef)]
    top = next(item for item in classes[0].body if isinstance(item, ast.FunctionDef))
    leaves = [next(x for x in cls.body if isinstance(x, ast.FunctionDef)) for cls in classes[1:]]
    operations, collective_order = Counter(), []
    shapes = Counter()
    for function in leaves:
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.match(r"(?:bf16|f32|i32|i64)\[", node.value):
                shapes[re.sub(r"\bs\d+\b", "S", node.value)] += 1
            if isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                if callee.startswith("torch."):
                    operations[callee] += 1
                if callee in {"torch.ops.vllm.all_reduce", "torch.ops.vllm.unified_attention_with_output"}:
                    collective_order.append((callee, [x.value for x in node.args if isinstance(x, ast.Constant)]))
    weights = [x.arg for x in top.args.args if "parameters_weight" in x.arg]
    depth_counts = Counter(re.search(r"campaign_execution_modules_(\d+)_", x).group(1) for x in weights if "campaign_execution_modules_" in x)
    return {
        "path": str(path), "sha256": digest(path.read_bytes()),
        "leaf_count": len(leaves), "top_weight_arguments": len(weights),
        "weight_depth_prefix_counts": dict(depth_counts),
        "operations": dict(operations), "shape_annotations": dict(shapes),
        "collective_attention_sequence_sha256": digest(json.dumps(collective_order)),
        "leaf_normalized_hashes": [digest(normalized_function(x)) for x in leaves],
    }


def artifact_summary(path):
    cache = ast.literal_eval(path.read_text())
    result = {}
    for key, metadata in sorted(cache.items()):
        artifact = Path(metadata["graph_handle"][1])
        assert artifact.parent == path.parent
        inductor_key, source = embedded_source(artifact)
        tree = ast.parse(source)
        call = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == "call")
        calls = [ast.unparse(x) for x in ast.walk(call) if isinstance(x, ast.Call)]
        kernels = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "triton":
                body = next(x.value for x in node.args[1:] if isinstance(x, ast.Constant) and isinstance(x.value, str))
                functions = [x for x in ast.walk(ast.parse(body)) if isinstance(x, ast.FunctionDef)]
                kernels.append(digest("\n".join(normalized_function(x) for x in functions)))
        result[str(key[1])] = {
            "path": str(artifact), "sha256": digest(artifact.read_bytes()),
            "inductor_key": inductor_key, "embedded_source_sha256": digest(source),
            "call_ast_sha256": digest(normalized_function(call)),
            "runtime_calls": calls, "triton_compute_hashes": kernels,
        }
    return result


def main():
    source = ROOT / "artifacts/storage_isolated_checkpoint3.json"
    output = []
    for record in json.loads(source.read_text())["records"]:
        for rank in range(record["P"]):
            arms = {}
            for arm in ["alias", "copies"]:
                base = Path(record[arm + "_path"]) / f"compile_cache/rank_{rank}_0/backbone"
                arms[arm] = {"graph": graph_summary(base / "computation_graph.py"), "artifacts": artifact_summary(base / "vllm_compile_cache.py")}
            a, b = arms["alias"], arms["copies"]
            ga, gb = a["graph"], b["graph"]
            artifact_keys = sorted(set(a["artifacts"]) | set(b["artifacts"]), key=int)
            comparisons = {}
            for key in artifact_keys:
                aa, bb = a["artifacts"].get(key), b["artifacts"].get(key)
                comparisons[key] = None if aa is None or bb is None else {
                    field + "_equal": aa[field] == bb[field]
                    for field in ["sha256", "embedded_source_sha256", "call_ast_sha256", "runtime_calls", "triton_compute_hashes"]
                }
            output.append({
                "model": record["model"], "P": record["P"], "rank": rank,
                "graph_operations_equal": ga["operations"] == gb["operations"],
                "leaf_shapes_equal": ga["shape_annotations"] == gb["shape_annotations"],
                "collective_attention_sequence_equal": ga["collective_attention_sequence_sha256"] == gb["collective_attention_sequence_sha256"],
                "different_normalized_leaf_indices": [i for i, pair in enumerate(zip(ga["leaf_normalized_hashes"], gb["leaf_normalized_hashes"])) if pair[0] != pair[1]],
                "artifact_comparisons": comparisons, "arms": arms,
            })
    if "--summary" in sys.argv:
        summarized = []
        for row in output:
            result = {key: value for key, value in row.items() if key != "arms"}
            result["graph_metadata"] = {
                arm: {key: row["arms"][arm]["graph"][key] for key in ["sha256", "leaf_count", "top_weight_arguments", "weight_depth_prefix_counts"]}
                for arm in ["alias", "copies"]
            }
            result["major_operations"] = {key: value for key, value in row["arms"]["alias"]["graph"]["operations"].items() if "linear" in key or "all_reduce" in key or "unified_attention" in key or "rms_norm" in key}
            summarized.append(result)
        output = summarized
    elif "--compact" in sys.argv:
        for row in output:
            for arm in ["alias", "copies"]:
                graph = row["arms"][arm]["graph"]
                row[arm + "_graph"] = {key: graph[key] for key in ["path", "sha256", "leaf_count", "top_weight_arguments", "weight_depth_prefix_counts"]}
            row["operations"] = row["arms"]["alias"]["graph"]["operations"]
            del row["arms"]
    print(json.dumps({"source": str(source), "source_sha256": digest(source.read_bytes()), "records": output}, indent=2))


if __name__ == "__main__":
    main()
