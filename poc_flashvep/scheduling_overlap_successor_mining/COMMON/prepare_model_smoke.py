"""Model-authored chat templates; explicit disabled Qwen3 thinking."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True, type=Path)
args = p.parse_args()
assert not args.out.exists()
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
questions = [
    ("smoke_arithmetic", "What is 6 times 7? Answer with only the number.", 16),
    ("smoke_capital", "What is the capital of France? Answer with only the city name.", 16),
    ("smoke_code", "Write a Python function that returns the square of its integer argument.", 64),
    ("smoke_reasoning", "Alice is older than Bob. Bob is older than Carol. Who is the youngest? Explain briefly.", 64),
]
rows = []
for i, (rid, text, length) in enumerate(questions):
    prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    rows.append({"request_id": rid, "prompt": prompt, "arrival_s": i,
                 "max_new_tokens": length, "ignore_eos": False,
                 "thinking": False, "prompt_tokens": len(tok.encode(prompt))})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
print(json.dumps({"requests": len(rows), "out": str(args.out),
                  "first_prompt": rows[0]["prompt"]}, indent=2))
