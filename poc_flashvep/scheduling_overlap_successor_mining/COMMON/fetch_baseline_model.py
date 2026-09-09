"""Pin/download an official baseline checkpoint without loading it on a GPU."""
import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    start = time.time()
    info = HfApi().model_info(args.repo)
    record = {"repo": args.repo, "revision": info.sha, "start_unix": start,
              "probe_only": args.probe_only}
    try:
        config = hf_hub_download(args.repo, "config.json", revision=info.sha)
        record["config"] = json.loads(Path(config).read_text())
        if not args.probe_only:
            record["snapshot"] = snapshot_download(
                args.repo, revision=info.sha, max_workers=4,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken",
                                "tokenizer*", "*.txt", "*.jinja"],
            )
        record["status"] = "PASS"
    except Exception as exc:
        record["status"] = "BLOCKED"
        record["error_type"] = type(exc).__name__
        # Do not print credentials, headers, or environment variables.
        record["http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
    record["elapsed_s"] = time.time() - start
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    if record["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
