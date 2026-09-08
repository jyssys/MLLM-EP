"""Preserve paper page boundaries and provenance in locally extracted text."""
import argparse
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("references", type=Path)
    args = ap.parse_args()
    records = []
    for path in sorted(args.references.glob("*.pdf")):
        reader = PdfReader(path)
        pages = [f"\n=== PAGE {i + 1} ===\n{page.extract_text()}\n"
                 for i, page in enumerate(reader.pages)]
        path.with_suffix(".txt").write_text("".join(pages))
        records.append({"file": path.name, "pages": len(pages),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.references / "paper_hashes.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
