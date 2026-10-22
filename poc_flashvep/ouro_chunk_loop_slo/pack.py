"""Lossless small-file archives. Never remove server originals."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

from .common import ROOT, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--unpack", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "raw_archive_manifest.json"
    if args.verify or args.unpack:
        data = json.loads(manifest.read_text())
        for entry in data["files"]:
            path = ROOT / entry["archive"]
            assert digest(path) == entry["archive_sha256"]
            plain = gzip.decompress(path.read_bytes())
            assert hashlib.sha256(plain).hexdigest() == entry["source_sha256"]
            if args.unpack:
                relative = Path(entry["source"])
                assert not relative.is_absolute() and ".." not in relative.parts
                target = ROOT / relative
                assert relative.parts[0] in ("raw", "logs")
                if target.exists():
                    assert digest(target) == entry["source_sha256"], (
                        "Do not overwrite different existing evidence"
                    )
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(plain)
        print("ARCHIVE_VERIFIED", len(data["files"]))
        return
    entries = []
    for folder in ("raw", "logs"):
        for source in sorted((ROOT / folder).rglob("*")):
            if not source.is_file() or source.name.endswith(".tmp"):
                continue
            relative = source.relative_to(ROOT)
            archive = ROOT / "raw_archive" / (str(relative) + ".gz")
            plain = source.read_bytes()
            buffer = io.BytesIO()
            with gzip.GzipFile(
                filename="", fileobj=buffer, mode="wb", compresslevel=6, mtime=0
            ) as output:
                output.write(plain)
            compressed = buffer.getvalue()
            assert len(compressed) < 95 * 1024**2, (
                "Split an oversized archive before git"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(compressed)
            entries.append(
                dict(
                    source=str(relative),
                    archive=str(archive.relative_to(ROOT)),
                    source_bytes=len(plain),
                    archive_bytes=len(compressed),
                    source_sha256=hashlib.sha256(plain).hexdigest(),
                    archive_sha256=hashlib.sha256(compressed).hexdigest(),
                )
            )
    write(
        manifest,
        dict(
            format="gzip per source file, deterministic mtime0, no original removal",
            files=entries,
            raw_bytes=sum(r["source_bytes"] for r in entries),
            archive_bytes=sum(r["archive_bytes"] for r in entries),
        ),
    )
    print("ARCHIVED", len(entries), "files")


if __name__ == "__main__":
    main()
