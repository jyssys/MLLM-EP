"""Independent HF full-prefix numerical reference, never a serving baseline.

One GPU (physical 4) holds the model. Native four-rank captures are compared on
the CPU after each full causal forward. Only the final generated-token positions
are projected to the vocabulary to avoid allocating prompt-wide logits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FASTPP"))
from run_configuration import assert_gpus_empty


def teacher_input(request):
    prompt, output = request["input_ids"], request["output_ids"]
    assert prompt and output
    # Last prompt token predicts output[0]; each following token predicts next.
    return prompt + output[:-1]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--screen", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert_gpus_empty()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True).eval()
    metadata = {"model": str(args.model), "physical_gpus": [4],
                "config_sha256": hashlib.sha256((args.model/"config.json").read_bytes()).hexdigest(),
                "torch": torch.__version__, "dtype": "float16", "runs": [],
                "purpose": "independent_HF_full_prefix_correctness_not_latency"}
    for screen in args.screen:
        manifest = json.loads((screen/"manifest.json").read_text())
        reference = json.loads(Path(manifest["reference"]).read_text())["requests"]
        count = len(reference)
        lengths = {len(r["output_ids"]) for r in reference}
        assert len(lengths) == 1
        generated = next(iter(lengths))
        histories = [teacher_input(r) for r in reference]
        assert len({len(ids) for ids in histories}) == 1
        inputs = torch.tensor(histories, dtype=torch.long, device="cuda:0")
        start = time.time()
        with torch.inference_mode():
            logits = model(input_ids=inputs, use_cache=False,
                           logits_to_keep=generated).logits.detach().cpu()
        end = time.time()
        assert logits.shape[:2] == (count, generated) and torch.isfinite(logits).all()
        target = args.out/screen.name/"reference_logits"
        target.mkdir(parents=True)
        for step in range(generated):
            torch.save({"step": step, "request_indices": list(range(2*count, 3*count)),
                        "logits": logits[:, step, :].clone(), "timing_excluded": True},
                       target/f"step{step:03d}.pt")
        row = {"screen": str(screen), "artifact": str(target), "requests": count,
               "positions": count*generated, "start_unix": start, "end_unix": end,
               "max_allocated_bytes": torch.cuda.max_memory_allocated(0)}
        metadata["runs"].append(row)
        (args.out/"manifest.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
