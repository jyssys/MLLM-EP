"""Archive, check, or restore explicit C07 fixture bytes. No Torch or CUDA import."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poc_ouro_parallelism_deep_discovery.fixture_archive import (
    create_archive, restore_archive, verify_archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    archive = commands.add_parser("archive", help="Keep the original and create a new chunk directory.")
    archive.add_argument("--fixture", type=Path, required=True)
    archive.add_argument("--archive-dir", type=Path)
    archive.add_argument("--chunk-mib", type=int, choices=range(1, 33), default=32)
    check = commands.add_parser("verify", help="Check all chunk hashes and the decoded byte stream.")
    check.add_argument("--manifest", type=Path, required=True)
    restore = commands.add_parser("restore", help="Reconstruct the exact bytes at a new output path.")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == "archive":
        result = create_archive(args.fixture, args.archive_dir, chunk_bytes=args.chunk_mib * 1024**2)
        directory = args.archive_dir or args.fixture.parent / "fixture.archive-v1"
        summary = dict(status=result["status"], manifest=str(directory.resolve() / "manifest.json"),
            original_bytes=result["original_bytes"], original_sha256=result["original_sha256"],
            compressed_stream_bytes=result["compressed_stream_bytes"], chunks=len(result["chunks"]),
            reconstruction_command=result["reconstruction"]["command"], original_retained=True)
    elif args.operation == "verify":
        summary = verify_archive(args.manifest)
    else:
        summary = restore_archive(args.manifest, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
