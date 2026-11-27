"""Fetch pinned Apache model data and verify every file against its Hub digest."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen


CHECKPOINTS = {
    "internvl3_8b": ("OpenGVLab/InternVL3-8B", "853e3a797a661694b1b8ece0cb72dc2b23e3dac9", 15_888_831_920),
    "qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 15_231_271_888),
}
HUB = "https://huggingface.co"


def download(item, base_url, output):
    name = item["rfilename"]
    path = output / name
    stage = path.with_suffix(path.suffix + ".partial")
    lfs = item.get("lfs")
    if not path.exists() and (not stage.exists() or stage.stat().st_size != item["size"]):
        subprocess.run([
            "curl", "--silent", "--show-error", "--fail", "--location", "--connect-timeout", "15",
            "--max-time", "2400", "--retry", "4", "--retry-delay", "2", "--continue-at", "-",
            "--output", str(stage), f"{base_url}/{name}?download=1",
        ], check=True)
    candidate = path if path.exists() else stage
    if candidate.stat().st_size != item["size"]:
        raise ValueError(f"Different file size for {name}")
    with candidate.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if lfs:
        expected = lfs["sha256"]
        computed = digest
    else:
        expected = item["blobId"]
        computed = hashlib.sha1(f"blob {item['size']}\0".encode() + candidate.read_bytes()).hexdigest()
    if computed != expected:
        raise ValueError(f"Different digest for {name}")
    if candidate == stage:
        stage.rename(path)
    print(f"Verified {name} ({item['size']:,} bytes)", flush=True)
    return {"path": name, "size": item["size"], "sha256": digest,
            "remote_digest": expected, "remote_type": "git-lfs-sha256" if lfs else "git-blob-sha1"}


def checkpoint_payload(output, index, group_by_module):
    dtype_bytes = Counter()
    module_bytes = Counter()
    inventory = index["weight_map"]
    seen = set()
    for shard in sorted(set(inventory.values())):
        with (output / shard).open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_size))
        header.pop("__metadata__", None)
        for name, tensor in header.items():
            if name in seen or inventory.get(name) != shard:
                raise ValueError(f"Unexpected tensor {name} in {shard}")
            seen.add(name)
            start, end = tensor["data_offsets"]
            group = name.split(".", maxsplit=1)[0] if group_by_module else "language_model"
            dtype_bytes[tensor["dtype"]] += end - start
            module_bytes[group] += end - start
    if seen != set(inventory) or dtype_bytes.total() != index["metadata"]["total_size"]:
        raise ValueError("Safetensors header payload and checkpoint index disagree")
    return {"by_dtype": dict(dtype_bytes), "by_module": dict(module_bytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=CHECKPOINTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--workers", default=5, type=int)
    args = parser.parse_args()
    model, revision, weight_bytes = CHECKPOINTS[args.checkpoint]
    with urlopen(f"{HUB}/api/models/{model}/revision/{revision}?blobs=true", timeout=30) as stream:
        manifest = json.load(stream)
    if manifest["sha"] != revision or manifest["gated"] or manifest["cardData"]["license"] != "apache-2.0":
        raise ValueError("Checkpoint metadata differs from the pinned public Apache-2.0 snapshot")
    items = [item for item in manifest["siblings"] if Path(item["rfilename"]).name == item["rfilename"]
             and not item["rfilename"].startswith(".")]
    shards = [item for item in items if item["rfilename"].endswith(".safetensors")]
    if len(shards) != 4 or sum(item["size"] for item in shards) != weight_bytes:
        raise ValueError("Unexpected safetensors inventory")
    args.output.mkdir(parents=True, exist_ok=True)
    base_url = f"{HUB}/{model}/resolve/{revision}"
    index = next(item for item in items if item["rfilename"] == "model.safetensors.index.json")
    download(index, base_url, args.output)
    safetensors_index = json.loads((args.output / index["rfilename"]).read_text())
    if set(safetensors_index["weight_map"].values()) != {item["rfilename"] for item in shards}:
        raise ValueError("Index and pinned safetensors inventory disagree")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, item, base_url, args.output) for item in items]
        files = [future.result() for future in futures]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({
        "model": model, "revision": revision, "license": "apache-2.0", "shards": len(shards),
        "weight_bytes": weight_bytes,
        "checkpoint_parameter_payload_bytes": checkpoint_payload(
            args.output, safetensors_index, group_by_module=args.checkpoint == "internvl3_8b",
        ),
        "files": files,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
