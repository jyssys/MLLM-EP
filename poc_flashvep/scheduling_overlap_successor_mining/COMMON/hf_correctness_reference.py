"""Same-checkpoint greedy correctness reference, not a serving speed baseline."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(8)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="balanced",
        attn_implementation="sdpa", local_files_only=True,
        max_memory={i: "70GiB" for i in range(4)},
    ).eval()
    rows = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    device = model.get_input_embeddings().weight.device
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inference_start = time.time()
    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False,
                                    max_new_tokens=args.max_new_tokens,
                                    repetition_penalty=1.0,
                                    return_dict_in_generate=True, output_scores=True)
        n = inputs.input_ids.shape[1]
        ids = output.sequences[0, n:].tolist()
        record = {"request_id": row["request_id"], "prompt_tokens": n,
                  "input_ids": inputs.input_ids[0].tolist(), "output_ids": ids,
                  "output_text": tokenizer.decode(ids, skip_special_tokens=True),
                  "raw_text": tokenizer.decode(ids, skip_special_tokens=False),
                  "first_top5_ids": output.scores[0][0].topk(5).indices.tolist(),
                  "first_top5_logits": output.scores[0][0].topk(5).values.float().tolist()}
        with args.out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    meta = {"model": args.model, "dtype": args.dtype, "elapsed_s": time.time()-start,
            "start_unix": start, "inference_start_unix": inference_start,
            "end_unix": time.time(), "repetition_penalty": 1.0,
            "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
            "torch": torch.__version__, "purpose": "correctness_only",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
