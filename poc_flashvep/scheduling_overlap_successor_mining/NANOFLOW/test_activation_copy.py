"""CPU AST seam: execute the actual copy_nano body, not a reimplementation.

The fake constructor avoids third-party import-time CUDA initialization. This
tests property preservation; full native output correctness is a separate gate.
"""
import argparse
import ast
from pathlib import Path


class ActivationFixture:
    def __init__(self, name, device, act_fn="silu_mul", nano_idx=None):
        self.name, self.device, self.act_fn = name, device, act_fn
        self.nano_ops = []
        self.category = "COMP"
        self.layer_list = [0, 1]
        self.N, self.tp_rank, self.tp_size = 1, 0, 1

    def set_category(self, category):
        self.category = category

    def expand_layer(self, layers):
        self.layer_list = layers

    def setShape(self, N, tp_rank, tp_size):
        self.N, self.tp_rank, self.tp_size = N, tp_rank, tp_size


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
args = parser.parse_args()
tree = ast.parse(args.source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Activation")
function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "copy_nano")
module = ast.Module(body=[function], type_ignores=[])
scope = {"Activation": ActivationFixture}
exec(compile(ast.fix_missing_locations(module), str(args.source), "exec"), scope)
for act_fn in ("sigmoid", "silu", "silu_mul"):
    original = ActivationFixture("SharedExpertActivation", "cpu", act_fn)
    copied = scope["copy_nano"](original, 0)
    assert copied.act_fn == original.act_fn, (original.act_fn, copied.act_fn)
    assert original.nano_ops == [copied]
print("ACTUAL_COPY_FACTORY_ACTIVATION_PRESERVATION_PASS: 3 activation types")
