"""Frozen equal-length real-text cohorts for the official Qwen1.5 MoE path."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--out", type=Path, required=True)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--prefill-only", action="store_true")
mode.add_argument("--high-decode", action="store_true")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
docs = [json.loads(s) for s in args.source.read_text().splitlines()]
random.Random(20260908).shuffle(docs)
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
encoded = [(row["id"], tokenizer.encode(row["text"], add_special_tokens=False)) for row in docs]
manifest = {"model": args.model, "source": str(args.source), "seed": 20260908,
            "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(), "traces": {}}
cases = ([("b4_c128", 4, 128), ("b64_c128", 64, 128), ("b256_c32", 256, 32)] if args.high_decode else
         [("b2_c128", 2, 128), ("b4_c256", 4, 256), ("b4_c1024", 4, 1024),
          ("b4_c2048", 4, 2048)] if args.prefill_only else
         [("b4_c128", 4, 128), ("b16_c512", 16, 512), ("b32_c128", 32, 128), ("b64_c128", 64, 128)])
for name, count, context in cases:
    rows = []
    for i in range(count):
        ids, provenance = [], []
        j = i * 7
        while len(ids) < context:
            source_id, values = encoded[j % len(encoded)]
            ids.extend(values)
            provenance.append(source_id)
            j += 1
        rows.append({"request_id": f"native_{i:03d}", "input_ids": ids[:context],
                     "max_new_tokens": 1 if args.prefill_only else 16 if args.high_decode else 32,
                     "ignore_eos": True, "source_ids": provenance})
    path = args.out / (name + ".jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["traces"][name] = {"requests": count, "context": context,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
