"""Freeze real Qwen image-processor tensors outside the Git worktree."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def prepare(manifest, model, out):
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True).image_processor
    rows = json.loads(manifest.read_text())["requests"]
    by_hash = {}
    for row in rows:
        for image in row["images"]:
            by_hash.setdefault(image["SHA256"], (row["request_id"], image))
    candidates = list(by_hash.values())
    chosen = []
    for target in (128, 384, 768, 1536, 3072, 5000):
        selected = min((item for item in candidates if item not in chosen),
                       key=lambda item: (abs(target - item[1]["visual_tokens"]), item[1]["SHA256"]))
        chosen.append(selected)

    def freeze(name, members):
        tensors, grids, items = [], [], []
        for request, item in members:
            path = Path(item["path"])
            if digest(path) != item["SHA256"]:
                raise ValueError(f"Original image digest changed for {request}")
            with Image.open(path) as image:
                prepared = processor(images=[image.convert("RGB")], return_tensors="pt")
            grid, = prepared["image_grid_thw"].tolist()
            if grid != item["grid_thw"]:
                raise ValueError(f"Actual image grid changed for {request}")
            pixels = prepared["pixel_values"].contiguous()
            if pixels.shape[0] != grid[0] * grid[1] * grid[2]:
                raise ValueError(f"Packed patch row count changed for {request}")
            tensors.append(pixels)
            grids.append(grid)
            items.append({"source": request, "image_sha256": item["SHA256"],
                          "raw_bytes": path.stat().st_size, "visual_tokens": item["visual_tokens"]})
        target = out/f"{name}.safetensors"
        save_file({"pixel_values": torch.cat(tensors), "grid": torch.tensor(grids)}, str(target))
        return {"id": name, "grids": grids, "items": items, "file_sha256": digest(target),
                "patch_rows": sum(int(t.shape[0]) for t in tensors), "processor_file_bytes": target.stat().st_size}

    out.mkdir(parents=True, exist_ok=False)
    entries = [freeze(f"image_{i}", [item]) for i, item in enumerate(chosen)]
    entries.append(freeze("group_left", chosen[:3]))
    entries.append(freeze("group_right", chosen[3:]))
    entries.append(freeze("group_all", chosen))
    (out/"manifest.json").write_text(json.dumps({"origin_manifest_sha256": digest(manifest),
        "model": str(model), "entries": entries}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.model, args.out)


if __name__ == "__main__":
    main()
