"""Publish the source fingerprints and compact, lossless EADA timing receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def describe(path, root):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": str(path.relative_to(root)), "sha256": digest, "bytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archives = []
    for name in args.cohorts:
        run = args.runs / name
        for path in sorted(run.glob("cohort_*.json")):
            archived = args.out / f"{name}_{path.name}.gz"
            archived.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            archive = describe(path, args.runs)
            archive["compressed_file"] = archived.name
            archive["config"] = describe(run / "config.json", args.runs)
            config = read(run / "config.json")
            archive["source_sha256"] = config["source_sha256"]
            archive["arrival_sha256"] = config["arrival_sha256"]
            archive["manifest_sha256"] = config["manifest_sha256"]
            archives.append(archive)
    leases = []
    for path in sorted(args.runs.glob("*/state/environment.json")):
        run = path.parent.parent
        environment = read(path)
        if "selected_uuids" not in environment:
            continue
        terminal = [describe(end, args.runs) for name in ("complete", "failure", "restore_failed")
                    if (end := path.parent / (name + ".json")).exists()]
        receipt = {"environment": describe(path, args.runs), "run": run.name,
                   "gpu_uuids": environment["selected_uuids"], "start_unix": environment["start_unix"],
                   "command": environment["command"], "terminal_files": terminal}
        receipt["terminal"] = [read(args.runs / item["file"]) for item in terminal]
        output = run / "logs" / "command.log"
        if output.exists():
            receipt["command_log"] = describe(output, args.runs)
        leases.append(receipt)
    witness_receipts = [describe(path, args.runs) for path in sorted(args.runs.glob("*witness*.json"))]
    drivers = [describe(path, args.runs) for path in sorted((args.runs / "drivers").glob("*.sh"))]
    (args.out / "manifest.json").write_text(json.dumps({
        "external_run_root": str(args.runs.resolve()), "cohorts": archives, "leases": leases,
        "witnesses": witness_receipts, "driver_snapshots": drivers,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
