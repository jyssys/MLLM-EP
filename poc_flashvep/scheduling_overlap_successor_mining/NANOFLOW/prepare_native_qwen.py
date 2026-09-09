"""CPU-only official weight packing for the supported Qwen1.5 MoE EP4 path."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from nanoflow.models.qwen2_moe.config_qwen2_moe import Qwen2MoEConfig
from nanoflow.models.qwen2_moe.qwen2_moe_ep import Pipeline


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    args.cache.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for rank in range(4):
        cfg = Qwen2MoEConfig(multi_gpu_mode=True, world_size=4, world_rank=rank,
                            ep_size=4, ep_rank=rank)
        cfg.cached_weight_dir = str(args.cache.resolve())
        binfile = args.cache / f"{cfg.cache_weight_name}_cuda:{rank}.bin"
        if binfile.exists():
            raise FileExistsError(f"Refusing silent reuse of incomplete cache: {binfile}")
        pipeline = Pipeline(cfg)
        pipeline.init_cached_weight(args.model)
        del pipeline
        print(json.dumps({"rank": rank, "elapsed_s": time.time()-start}), flush=True)
    (args.cache / "PACKING_COMPLETE.json").write_text(json.dumps({
        "model": args.model, "ep_size": 4, "dtype": "float16",
        "source": "official dev-h100 Pipeline.init_cached_weight",
        "elapsed_s": time.time()-start,
        "cuda_initialized": torch.cuda.is_initialized(),
    }, indent=2))


if __name__ == "__main__":
    main()
