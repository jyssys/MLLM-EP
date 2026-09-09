"""Make page-labelled local text for full-paper audit (no GPU imports)."""
import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = []
    for path in sorted(args.directory.glob("*.pdf")):
        reader = PdfReader(path)
        text = "\n\n".join(
            f"=== PDF PAGE {i + 1} ===\n{page.extract_text()}"
            for i, page in enumerate(reader.pages)
        )
        path.with_suffix(".txt").write_text(text)
        manifest.append({"file": path.name, "pages": len(reader.pages),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "characters": len(text)})
    (args.directory / "pdf_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
