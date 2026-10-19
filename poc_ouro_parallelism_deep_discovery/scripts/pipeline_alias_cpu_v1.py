"""CPU toy for pipeline IR storage. No model weights or distributed workers.

Run with CUDA_VISIBLE_DEVICES empty. The optional alias rebinding changes only
the new toy objects. It does not change installed package code.
"""

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES to an empty string for this CPU control.")

import torch
from torch import nn
from torch.distributed.pipelining import pipeline, pipe_split
import torch.distributed.pipelining._IR as ir


ROOT = Path(__file__).resolve().parents[1]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NonlinearBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        matrix = (torch.arange(16, dtype=torch.float64).reshape(4, 4) - 7.5) / 64
        matrix = matrix + torch.eye(4, dtype=torch.float64) * (0.5 + index / 32)
        bias = torch.tensor([-3, -1, 1, 3], dtype=torch.float64) * (index + 1) / 128
        self.weight = nn.Parameter(matrix, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.tanh(torch.nn.functional.linear(x, self.weight, self.bias)) + x / 8


class FourPassCore(nn.Module):
    def __init__(self, split_kind):
        super().__init__()
        self.core = nn.ModuleList([NonlinearBlock(i) for i in range(4)])
        self.split_kind = split_kind

    def forward(self, x):
        for depth in range(4):
            for layer in range(4):
                x = self.core[layer](x)
                final = depth == 3 and layer == 3
                if not final and self.split_kind == "physical":
                    pipe_split()
            if depth != 3 and self.split_kind == "depth":
                pipe_split()
        return x


def inventory(items):
    rows = []
    for name, parameter in items:
        storage = parameter.untyped_storage()
        rows.append({
            "name": name, "python_object_id": id(parameter),
            "storage_data_ptr": storage.data_ptr(), "tensor_data_ptr": parameter.data_ptr(),
            "storage_offset": parameter.storage_offset(), "storage_nbytes": storage.nbytes(),
            "shape": list(parameter.shape), "stride": list(parameter.stride()),
            "dtype": str(parameter.dtype), "device": str(parameter.device),
            "requires_grad": parameter.requires_grad,
        })
    unique_storage = {r["storage_data_ptr"]: r["storage_nbytes"] for r in rows}
    return {"references": rows, "reference_count": len(rows),
            "unique_parameter_objects": len({r["python_object_id"] for r in rows}),
            "unique_storages": len(unique_storage), "unique_storage_bytes": sum(unique_storage.values())}


def module_inventory(module):
    return inventory(module.named_parameters(remove_duplicate=False))


def output_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    assert isinstance(value, tuple) and len(value) == 1
    return value[0]


def stage_items(pipe):
    result = []
    for i in range(pipe.num_stages):
        for name, parameter in pipe.get_stage_module(i).named_parameters(remove_duplicate=False):
            result.append((f"stage{i}.{name}", parameter))
    return result


def active_stage_items(pipe):
    """Collect parameter reads from reachable nodes in this toy's stage graphs."""
    result = []

    def visit(module, prefix):
        if not hasattr(module, "graph"):
            result.extend((prefix + name, p) for name, p in module.named_parameters(remove_duplicate=False))
            return
        for node in module.graph.nodes:
            if node.op == "get_attr":
                value = module
                for atom in node.target.split("."):
                    value = getattr(value, atom)
                if isinstance(value, nn.Parameter):
                    result.append((prefix + node.target, value))
            elif node.op == "call_module":
                visit(module.get_submodule(node.target), prefix + node.target + ".")

    for i in range(pipe.num_stages):
        visit(pipe.get_stage_module(i), f"stage{i}.")
    return result


def rebind_same_physical_parameters(pipe):
    """A local manual control, not stock distributed pipeline behavior."""
    canonical = {}
    replacements = []
    for i in range(pipe.num_stages):
        stage = pipe.get_stage_module(i)
        for name, parameter in list(stage.named_parameters(remove_duplicate=False)):
            physical_name = re.sub(r"@\d+", "", name)
            canonical.setdefault(physical_name, parameter)
            parent, _, leaf = name.rpartition(".")
            target = stage.get_submodule(parent) if parent else stage
            setattr(target, leaf, canonical[physical_name])
            replacements.append({"stage": i, "parameter": name,
                                 "physical_parameter": physical_name,
                                 "conceptual_owner": int(physical_name.split(".")[1])})
    return replacements


def run_arm(kind, x, expected):
    model = FourPassCore(kind).eval()
    original = module_inventory(model)
    unrolled_view = nn.ModuleList([model.core[layer] for depth in range(4) for layer in range(4)])
    aliased_unroll = module_inventory(unrolled_view)
    exported = torch.export.export(model, (x,))
    exported_state = inventory(exported.state_dict.items())
    exported_output = exported.module()(x)
    pipe = pipeline(model, mb_args=(x,))
    converted = inventory(stage_items(pipe))
    active = inventory(active_stage_items(pipe))
    actual = output_tensor(pipe(x))
    old_ptrs = {r["storage_data_ptr"] for r in original["references"]}
    new_ptrs = {r["storage_data_ptr"] for r in active["references"]}
    result = {
        "split_kind": kind, "num_stages": pipe.num_stages,
        "source_inventory": original, "aliased_unroll_inventory": aliased_unroll,
        "exported_state_inventory": exported_state,
        "default_pipe_registered_inventory": converted,
        "default_pipe_active_read_inventory": active,
        "default_pipe_active_original_shared_storage_count": len(old_ptrs & new_ptrs),
        "replicated_params": pipe.replicated_params,
        "top_level_graph": str(pipe.split_gm.graph),
        "exported_output_equal": torch.equal(exported_output, expected),
        "pipe_output_equal": torch.equal(actual, expected),
        "max_abs_error": float((actual - expected).abs().max()),
        "output": actual.detach().tolist(),
    }
    assert original["unique_storage_bytes"] == aliased_unroll["unique_storage_bytes"] == 640
    assert original["unique_storages"] == aliased_unroll["unique_storages"] == 8
    assert active["unique_storage_bytes"] == 2560 and active["unique_storages"] == 32
    assert converted["unique_storage_bytes"] >= active["unique_storage_bytes"]
    assert len(old_ptrs & new_ptrs) == 0
    assert result["exported_output_equal"] and result["pipe_output_equal"]
    if kind == "physical":
        result["manual_alias_rebindings"] = rebind_same_physical_parameters(pipe)
        result["manual_alias_inventory"] = inventory(stage_items(pipe))
        result["manual_alias_active_inventory"] = inventory(active_stage_items(pipe))
        result["manual_alias_output_equal"] = torch.equal(output_tensor(pipe(x)), expected)
        result["manual_alias_scope"] = "All stages remain on CPU. Owner labels are conceptual. No distributed cost or worker execution."
        assert result["manual_alias_inventory"]["unique_storage_bytes"] == 640
        assert result["manual_alias_active_inventory"]["unique_storage_bytes"] == 640
        assert result["manual_alias_output_equal"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = utc()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    x = torch.tensor([[-0.75, -0.25, 0.5, 1.0], [0.125, 0.375, -0.5, 0.875]], dtype=torch.float64)
    expected = FourPassCore("none").eval()(x).detach()
    arms = [run_arm(kind, x, expected) for kind in ["depth", "physical"]]
    assert not torch.cuda.is_initialized()
    result = {
        "schema_version": 1, "start_UTC": start, "end_UTC": utc(),
        "evidence_class": "CPU_TOY_IR_STORAGE_CONTROL",
        "scope": "Four deterministic nonlinear physical blocks, each visited four times. No research model, KV, gates, CUDA, distributed workers, or latency benchmark.",
        "command": "CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/cp-composition-vllm026/bin/python scripts/pipeline_alias_cpu_v1.py --output artifacts/pipeline_alias_cpu_v1.json",
        "python": sys.version, "python_executable": sys.executable,
        "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
        "IR_path": inspect.getfile(ir), "IR_sha256": sha(inspect.getfile(ir)),
        "script_sha256": sha(__file__), "CUDA_initialized_after_test": torch.cuda.is_initialized(),
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64 CPU", "input": x.tolist(), "reference_output": expected.tolist(),
        "arms": arms,
        "initial_control_failure": "An initial assertion assumed registered storage equaled four active copies. It failed. The corrected test records reachable parameter reads separately from unused registered parameters.",
        "limitations": ["No BF16 or Ouro numerical claim.", "Pointer identities are local to this process.",
                        "Manual alias rebinding is an explicit toy operation, not a package capability claim.",
                        "No communication, scheduling, ownership protocol, or HBM cost measurement."],
    }
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "artifacts")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"start_UTC": start, "end_UTC": result["end_UTC"],
                      "torch_version": result["torch_version"], "IR_sha256": result["IR_sha256"],
                      "CUDA_initialized": result["CUDA_initialized_after_test"],
                      "arms": [{"kind": a["split_kind"], "stages": a["num_stages"],
                                "original_bytes": a["source_inventory"]["unique_storage_bytes"],
                                "unroll_bytes": a["aliased_unroll_inventory"]["unique_storage_bytes"],
                                "export_bytes": a["exported_state_inventory"]["unique_storage_bytes"],
                                "pipe_registered_bytes": a["default_pipe_registered_inventory"]["unique_storage_bytes"],
                                "pipe_active_bytes": a["default_pipe_active_read_inventory"]["unique_storage_bytes"],
                                "output_equal": a["pipe_output_equal"],
                                "manual_alias_bytes": a.get("manual_alias_inventory", {}).get("unique_storage_bytes")}
                               for a in arms]}, indent=2))


if __name__ == "__main__":
    main()
