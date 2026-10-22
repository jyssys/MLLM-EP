"""Lossless Git-sized raw archives. Original server files remain untouched."""

import argparse
import gzip
import hashlib
import json
from .common import ROOT, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    if a.restore:
        for row in json.loads((ROOT / "archive_manifest.json").read_text()):
            target = (ROOT / row["source"]).resolve()
            assert target.is_relative_to(ROOT.resolve())
            data = gzip.decompress((ROOT / row["archive"]).read_bytes())
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest() == row["sha256"]
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        return
    rows = []
    files = list((ROOT / "raw").glob("*/*.json"))
    files += [
        ROOT / f
        for f in (
            "request_events.jsonl",
            "scheduler_quantum_trace.jsonl",
            "gpu_timeline.jsonl",
            "nsys_kernel_timeline.csv",
            "nsys_copy_timeline.csv",
            "nsys_runtime_timeline.csv",
        )
        if (ROOT / f).exists()
    ]
    for source in files:
        data = source.read_bytes()
        relative = source.relative_to(ROOT)
        target = ROOT / "raw_archive" / (str(relative) + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        compressed = gzip.compress(data, compresslevel=9, mtime=0)
        assert gzip.decompress(compressed) == data
        target.write_bytes(compressed)
        rows.append(
            dict(
                source=str(relative),
                archive=str(target.relative_to(ROOT)),
                bytes=len(data),
                compressed_bytes=len(compressed),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    write(ROOT / "archive_manifest.json", rows)
    print(
        "ARCHIVED",
        len(rows),
        "files",
        sum(r["bytes"] for r in rows),
        "->",
        sum(r["compressed_bytes"] for r in rows),
        "bytes",
    )


if __name__ == "__main__":
    main()
