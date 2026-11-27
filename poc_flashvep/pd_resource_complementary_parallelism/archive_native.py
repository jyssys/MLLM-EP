"""Package native measurements without model data, request inputs, or runtime credentials."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile


# The traffic payloads and prompt token IDs live outside this archive. Client
# JSONL contains response token IDs and timing, Prometheus captures counters.
RECEIPTS = {"run.json", "requests.jsonl", "queues.jsonl", "gpu.jsonl", "shapes.json",
            "start_profile.json", "stop_profile.json", "start_profile_clock.json",
            "stop_profile_clock.json", "warmup.json", "kv_layout.json", "protocol.txt", "exit_codes.txt"}
ENGINE_LOG = re.compile(r"(?:PREF|P|D|R)\d+\.log")


def allowed(source, full):
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(source)
        if any(part.startswith(".") for part in rel.parts):
            continue
        name = path.name
        if (name in RECEIPTS or name.endswith((".prom", "_command.json", "_gpu.csv")) or ENGINE_LOG.fullmatch(name)
                or ("activity" in rel.parts and name.startswith("worker-") and path.suffix in {".json", ".csv"})
                or (full and "profiles" in rel.parts and name.endswith(".pt.trace.json.gz"))):
            yield rel


def pack(source, groups, target, include_kineto):
    files = []
    with target.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for group in groups:
                for relative in allowed(source / group, include_kineto):
                    name = Path(group) / relative
                    path = source / name
                    info = archive.gettarinfo(str(path), arcname=str(name))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as payload:
                        archive.addfile(info, payload)
                        payload.seek(0)
                        file_digest = hashlib.file_digest(payload, "sha256").hexdigest()
                    files.append({"path": str(name), "bytes": path.stat().st_size, "sha256": file_digest})
    with target.open("rb") as result:
        digest = hashlib.file_digest(result, "sha256")
    target.with_name(target.name + ".manifest.json").write_text(
        json.dumps({"archive_sha256": digest.hexdigest(), "included_files": files}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--include-kineto", action="store_true")
    args = parser.parse_args()
    for group in args.groups:
        if Path(group).name != group or not (args.raw_root / group).is_dir():
            parser.error(f"Unknown top-level group {group!r}")
    pack(args.raw_root, args.groups, args.out, args.include_kineto)


if __name__ == "__main__":
    main()
