"""Use FastPP's exact official Azure sampling function and record its semantics.

This trace is an official traffic/length reproduction control, not real natural
prompt text. The common FineWeb traces provide the latter separately.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from sglang.bench_serving import sample_azure_trace_2023_requests

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--requests", type=int, default=192)
p.add_argument("--timestamp-speedup", type=float, default=0.9)
args = p.parse_args()
assert not args.out.exists()
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
random.seed(20260908)
url = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv"
cache_name = "scheduling_successor_azure_conv_2023_ts.txt"
requests = sample_azure_trace_2023_requests(
    url, cache_name, args.requests, tok, include_timestamps=True)
origin = requests[0][3]
rows, excluded = [], []
for i, (prompt, declared_m, output, timestamp) in enumerate(requests):
    actual_m = len(tok.encode(prompt, add_special_tokens=False))
    if actual_m + output > 16384 or actual_m == 0 or output <= 0:
        excluded.append({"index": i, "actual_m": actual_m, "output": output})
        continue
    rows.append({"request_id": f"azure_conv_{i:04d}",
                 "arrival_s": (timestamp-origin)/args.timestamp_speedup,
                 "prompt": prompt, "prompt_tokens": actual_m,
                 "official_declared_prompt_tokens": declared_m,
                 "max_new_tokens": output, "ignore_eos": True,
                 "prompt_semantics": "official uniformly random token IDs decoded to string",
                 "source_timestamp_s": timestamp, "modality": "text"})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text("".join(json.dumps(row)+"\n" for row in rows))
meta = {"url": url, "seed": 20260908, "requests": len(rows),
        "excluded_context_overflow_or_empty": excluded,
        "timestamp_speedup": args.timestamp_speedup,
        "trace_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "official_cache_sha256": hashlib.sha256((Path("/tmp")/cache_name).read_bytes()).hexdigest(),
        "retokenization_length_changes": sum(r["prompt_tokens"] != r["official_declared_prompt_tokens"] for r in rows),
        "last_arrival_s": rows[-1]["arrival_s"],
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_output_tokens": sum(r["max_new_tokens"] for r in rows),
        "function": "official sglang.bench_serving.sample_azure_trace_2023_requests"}
args.out.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
