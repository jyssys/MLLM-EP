"""Download and verify the official pinned ERNIE-VL snapshot outside Git."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


CHECKPOINT = "baidu/ERNIE-4.5-VL-28B-A3B-Thinking"
REVISION = "e7487b6ff5e726e21bb8bcf9af83b34a82fcdaae"
BASE = "https://huggingface.co"
SHARD_TOTAL_BYTES = 59_347_186_416


def download(original, target):
    filename = original["rfilename"]
    path = target / filename
    staged = path.with_suffix(path.suffix + ".partial")
    url = f"{BASE}/{CHECKPOINT}/resolve/{REVISION}/{filename}"
    if not path.exists():
        if filename.endswith(".safetensors"):
            if not staged.exists() or staged.stat().st_size != original["size"]:
                subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--connect-timeout", "15",
                                "--retry", "4", "--retry-delay", "2", "--continue-at", "-", "--output", str(staged),
                                f"{url}?download=1"], check=True)
        else:
            with urllib.request.urlopen(url, timeout=30) as stream:
                staged.write_bytes(stream.read(original["size"] + 1))
    candidate = path if path.exists() else staged
    length = candidate.stat().st_size
    if length != original["size"]:
        raise ValueError(f"{filename} size {length} differs from pinned size {original['size']}")
    with candidate.open("rb") as source:
        local_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if lfs := original.get("lfs"):
        digest_type, remote, local = "git-lfs-sha256", lfs["sha256"], local_sha256
    else:
        digest_type, remote = "git-blob-sha1", original["blobId"]
        local = hashlib.sha1(f"blob {length}\0".encode() + candidate.read_bytes()).hexdigest()
    if local != remote:
        raise ValueError(f"{filename} {digest_type} {local} differs from pinned digest {remote}")
    if not path.exists():
        staged.rename(path)
    print(f"Verified {filename} ({length:,} bytes)", flush=True)
    return {"path": filename, "size": length, "sha256": local_sha256,
            "remote_digest_type": digest_type, "remote_digest": remote}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with urllib.request.urlopen(f"{BASE}/api/models/{CHECKPOINT}/revision/{REVISION}?blobs=true", timeout=30) as response:
        manifest = json.load(response)
    if manifest["sha"] != REVISION or manifest["gated"] or manifest["cardData"]["license"] != "apache-2.0":
        raise ValueError("Checkpoint release no longer matches the frozen provenance")
    args.output.mkdir(exist_ok=True, parents=True)
    selected = [row for row in manifest["siblings"] if row["rfilename"] not in {".gitattributes", "benchmark.jpg"}]
    if any(Path(row["rfilename"]).name != row["rfilename"] for row in selected):
        raise ValueError("Expected top-level snapshot files")
    shards = [row for row in selected if row["rfilename"].endswith(".safetensors")]
    if len(shards) != 29 or sum(row["size"] for row in shards) != SHARD_TOTAL_BYTES:
        raise ValueError("Pinned shard inventory changed")
    index = next(row for row in selected if row["rfilename"] == "model.safetensors.index.json")
    download(index, args.output)
    weight_map = json.loads((args.output / index["rfilename"]).read_text())["weight_map"]
    if set(weight_map.values()) != {row["rfilename"] for row in shards}:
        raise ValueError("Pinned index and shard inventory differ")
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        files = list(workers.map(partial(download, target=args.output), selected))
    result = {"checkpoint": CHECKPOINT, "revision": REVISION, "license": "apache-2.0", "gated": False,
              "shard_count": len(shards), "shard_total_bytes": SHARD_TOTAL_BYTES, "files": files}
    args.receipt.parent.mkdir(exist_ok=True, parents=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
