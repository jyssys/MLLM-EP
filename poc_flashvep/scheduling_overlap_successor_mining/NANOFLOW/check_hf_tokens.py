"""Compare every native smoke token to the same-checkpoint FP16 HF reference."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--native", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
native = json.loads(args.native.read_text())
reference = json.loads(args.reference.read_text().splitlines()[0])
checks = []
for i, ids in enumerate(native["token_ids_including_prompt"]):
    prompt = reference["input_ids"]
    generated = ids[len(prompt):]
    expected = reference["output_ids"][:len(generated)]
    checks.append({"request_index": i, "tokens": len(generated),
                   "prompt_exact": ids[:len(prompt)] == prompt,
                   "generated_exact": generated == expected,
                   "first_difference": next((j for j, (a, b) in
                       enumerate(zip(generated, expected)) if a != b), None)})
meta_path = args.native.with_suffix(".run_meta.json")
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
summary = {"status": "PASS" if all(c["prompt_exact"] and c["generated_exact"] for c in checks) else "FAIL",
           "checks": checks, "total_compared_output_tokens": sum(c["tokens"] for c in checks),
           "manual_plan_parts": meta.get("manual_plan_parts"),
           "cuda_graph": meta.get("fixed_decode_cuda_graph"),
           "scope": "four-request official smoke under the recorded plan only; not arbitrary shapes"}
args.out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
assert summary["status"] == "PASS"
