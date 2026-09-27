from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_vision_conditioned_decode_reduction.track_a import (
    audit_existing_evidence,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_existing_evidence(args.prior_root)
    _atomic_json(args.output_dir / "vector_availability.json", audit)
    print(json.dumps({
        "audit_status": audit["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
