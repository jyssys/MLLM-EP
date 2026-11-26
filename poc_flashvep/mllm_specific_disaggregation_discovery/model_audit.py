"""Record frozen model and implementation identity without initializing CUDA."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot

MODULES = {
    "transformers": (
        "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
        "transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py",
        "transformers/cache_utils.py",
    ),
    "vllm": (
        "vllm/config/model.py",
        "vllm/config/parallel.py",
        "vllm/model_executor/models/qwen2_5_vl.py",
        "vllm/v1/attention/backends/flash_attn.py",
        "vllm/vllm_flash_attn/flash_attn_interface.py",
        "vllm/v1/attention/ops/merge_attn_states.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    rt.require(args.model.name == rt.MODEL_REVISION, "Unexpected model revision")
    sources = {}
    for package, paths in MODULES.items():
        dist = importlib.metadata.distribution(package)
        for path in paths:
            origin = dist.locate_file(path)
            sources[path] = {"SHA256": rt.sha(origin), "bytes": origin.stat().st_size}
    rt.write(args.out, {
        "model_revision": rt.MODEL_REVISION, "model_config": config,
        "model_file_identities": {
            path.name: rt.sha(path) for path in args.model.iterdir()
            if path.is_file() and path.suffix == ".json"
        },
        "runtime": rt.runtime_inventory(), "source_files": sources,
        "hardware": {
            "gpu_csv": rt.nvidia("--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id"),
            "platform": {key: value["stdout"] for key, value in snapshot(topology=True).items()
                         if key in ("topology", "cpu", "disk", "numa")},
        },
    })


if __name__ == "__main__":
    main()
