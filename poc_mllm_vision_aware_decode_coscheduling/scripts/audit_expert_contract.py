from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from poc_mllm_vision_aware_decode_coscheduling.contract import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = (
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    )
    tensors = {}
    bytes_per_expert = 0
    for name in names:
        path = args.snapshot / index[name]
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_slice(name)
            shape = [int(value) for value in tensor.get_shape()]
            dtype = str(tensor.get_dtype())
        if shape[0] != 128 or dtype != "BF16":
            raise RuntimeError((name, shape, dtype))
        bytes_per_expert += int(shape[1] * shape[2] * 2)
        tensors[name] = {"shape": shape, "dtype": dtype, "file": path.name}
    atomic_json(
        args.output,
        {
            "tensors": tensors,
            "bytes_per_expert_per_layer": bytes_per_expert,
            "mib_per_expert_per_layer": bytes_per_expert / (1 << 20),
            "label": "VERIFIED_CHECKPOINT_WEIGHT_BYTES_NOT_HBM_TRAFFIC",
        },
    )


if __name__ == "__main__":
    main()
