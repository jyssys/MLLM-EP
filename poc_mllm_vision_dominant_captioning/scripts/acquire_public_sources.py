from __future__ import annotations

from huggingface_hub import hf_hub_download

FILES = (
    ("not-lain/docci-small", "d3709dca05c2a9c1e16f09e513e943b25047be9d", "data/train-00000-of-00001.parquet"),
    ("lmms-lab-encoder/TextCaps", "e9e223338832318a6161d04648871c18a024ce85", "data/val-00000-of-00002.parquet"),
    ("lmms-lab-encoder/NoCaps", "a26b3fe1e0021164ec430c57c48085d58f2fe922", "data/validation-00000-of-00003-55b5edbf01d47b39.parquet"),
)


def main() -> None:
    for repo, revision, filename in FILES:
        print(hf_hub_download(repo_id=repo, revision=revision, filename=filename, repo_type="dataset"))


if __name__ == "__main__":
    main()
