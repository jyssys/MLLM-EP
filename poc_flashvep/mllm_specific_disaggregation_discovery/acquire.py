"""Fetch the frozen benchmark shards and verify their publisher LFS digests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

from .dataset import PINS, REPOS, file_hash, require, write


def expected_shards(study_source, revision):
    def selected(filename):
        if study_source == "MMMU":
            return filename.endswith("/validation-00000-of-00001.parquet")
        if study_source == "MathVista":
            return filename.startswith("data/testmini") and filename.endswith(".parquet")
        return filename == "testmini-mmreason-00000-of-00001.parquet"

    repo = REPOS[study_source]
    url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as stream:
        metadata = json.load(stream)
    require(metadata["sha"] == revision, f"Publisher revision differs for {repo}")
    shards = [field for field in metadata["siblings"] if selected(field["rfilename"])]
    require(len(shards) == (30 if study_source == "MMMU" else 1), f"Unexpected shard roster for {repo}")
    return [(repo, revision, field) for field in shards]


def acquire(output):
    from huggingface_hub import hf_hub_download

    source_dir = output / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    def fetch(job):
        repo, revision, field = job
        filename = field["rfilename"]
        local = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename, revision=revision,
                                    local_dir=source_dir / repo, token=False))
        sha = file_hash(local)
        size = local.stat().st_size
        require(field.get("size") == size and field.get("lfs", {}).get("sha256") == sha,
                f"Published LFS hash or length differs for {repo}/{filename}")
        return {"repo_id": repo, "revision": revision, "filename": filename, "sha256": sha, "bytes": size}

    jobs = [job for source, (revision, _, _) in PINS.items() for job in expected_shards(source, revision)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, jobs))
    write(source_dir / "receipt.json", receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.output.resolve())


if __name__ == "__main__":
    main()
