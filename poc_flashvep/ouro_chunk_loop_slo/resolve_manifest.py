"""Append explicitly registered diagnostic jobs without changing frozen cases."""

import argparse
import json
from pathlib import Path

from .common import ROOT, digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    original = Path(args.manifest).resolve()
    registry = ROOT / "manifest_extensions.json"
    additions = (
        json.loads(registry.read_text()).get(original.name, [])
        if registry.exists()
        else []
    )
    if not additions:
        print(original)
        return
    source = json.loads(original.read_text())
    jobs = list(source["jobs"])
    provenance = [dict(path=str(original), sha256=digest(original))]
    for name in additions:
        addition = ROOT / name
        jobs.extend(json.loads(addition.read_text())["jobs"])
        provenance.append(dict(path=str(addition), sha256=digest(addition)))
    assert len({j["id"] for j in jobs}) == len(jobs), "Duplicate job ID"
    data = dict(
        purpose="Explicit extension of an immutable base queue. No original case changes.",
        sources=provenance,
        jobs=jobs,
    )
    # The digest preserves earlier resolved queues when diagnostic extensions grow.
    import hashlib

    suffix = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
    destination = ROOT / (original.stem + "_resolved_" + suffix + ".json")
    if destination.exists():
        assert json.loads(destination.read_text()) == data, (
            "Queue changed after resolution"
        )
    else:
        write(destination, data)
    print(destination)


if __name__ == "__main__":
    main()
