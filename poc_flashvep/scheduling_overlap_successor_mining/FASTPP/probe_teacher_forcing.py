"""Separate tokenization/sampling from cached decode correctness. CPU client."""
import argparse
import json
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--url", default="http://127.0.0.1:31800")
args = parser.parse_args()
assert not args.out.exists()
rows = [json.loads(x) for x in args.trace.read_text().splitlines()]
cases = [("free_arithmetic", rows[0]["prompt"], 5),
         ("forced_4", rows[0]["prompt"] + "4", 1),
         ("forced_42", rows[0]["prompt"] + "42", 1),
         ("free_capital", rows[1]["prompt"], 5),
         ("forced_Paris", rows[1]["prompt"] + "Paris", 1)]
for name, prompt, length in cases:
    payload = {"text": prompt, "sampling_params": {
        "temperature": 0, "max_new_tokens": length, "ignore_eos": False,
        "skip_special_tokens": False}, "stream": False}
    request = urllib.request.Request(args.url + "/generate",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    row = {"case": name, "prompt": prompt, "response": result}
    with args.out.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
