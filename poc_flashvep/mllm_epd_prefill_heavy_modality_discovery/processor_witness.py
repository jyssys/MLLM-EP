"""Rerun the pinned multimodal processor on representatives of frozen workloads."""

import argparse
from pathlib import Path
import time

from .datasets import digest, ordered_hash, read, write


def inspect(processor, row):
    import numpy as np
    from PIL import Image

    pixels = []
    stored = row["images"]
    for item in stored:
        with Image.open(item["path"]) as image:
            if list(image.size) != item["original_resolution"]:
                raise ValueError(f"Source dimensions changed at {item['path']}")
            pixels.append(image.convert("RGB"))
    try:
        start = time.perf_counter()
        result = processor(text=[row["prompt"]], images=pixels or None, return_tensors="pt")
        observed_s = time.perf_counter() - start
    finally:
        for image in pixels:
            image.close()
    ids = result["input_ids"].numpy().reshape(-1)
    grids = result["image_grid_thw"].tolist() if stored else []
    visual = int((ids == processor.image_token_id).sum())
    counts_match = [len(ids), visual, grids] == [row["total_input_tokens"], row["visual_tokens"], row["grid_thw"]]
    if "token_metadata_file" in row:
        with np.load(row["token_metadata_file"], allow_pickle=False) as data:
            ids_match = np.array_equal(ids, data["input_ids"].reshape(-1))
    else:
        ids_match = digest(np.asarray(ids, dtype="<i8").tobytes()) == row["full_input_ids_SHA256"]
    if not ids_match or not counts_match:
        raise ValueError(f"Processor witness mismatch for {row['request_id']}: {len(ids)} rows, {visual} visual")
    tensor = result.get("pixel_values")
    shapes = {"processor_tensor_dtype": str(tensor.dtype) if tensor is not None else None,
              "processor_tensor_shape": list(tensor.shape) if tensor is not None else [0],
              "processor_tensor_bytes": int(tensor.numel() * tensor.element_size()) if tensor is not None else 0}
    return {"request_id": row["request_id"], "images": len(stored), "input_tokens": len(ids),
            "visual_tokens": visual, "vision_patches": sum(t*h*w for t,h,w in grids),
            "source_image_bytes": sum(Path(item["path"]).stat().st_size for item in stored),
            **shapes,
            "logical_bf16_lm_ready_image_embedding_bytes": visual * 3584 * 2,
            "logical_bf16_full_prompt_kv_bytes": len(ids) * 28 * 2 * 4 * 128 * 2,
            "grid_parity_pass": counts_match, "input_token_identity_pass": ids_match,
            "processor_only_wall_s_diagnostic": observed_s}


def choose(rows, per_stratum):
    ordered = sorted(rows, key=ordered_hash)
    if not per_stratum:
        return ordered[:1]
    groups = {}
    for row in ordered:
        groups.setdefault(row["stratum"], row)
    return list(groups.values())


def main():
    import torch
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    witnesses = {}
    for manifest in (args.natural, *args.controlled):
        selected = choose(read(manifest)["requests"], per_stratum=manifest == args.natural)
        witnesses[manifest.name] = []
        for row in selected:
            entry = inspect(processor, row)
            witnesses[manifest.name].append(entry)
            print(f"{row['request_id']} exact processor input, {entry['input_tokens']} rows", flush=True)
    write(args.out, {"evidence":"CORRECTNESS_ONLY", "scope":"actual pinned CPU processor and input token identity",
                     "witnesses":witnesses})


if __name__ == "__main__":
    main()
