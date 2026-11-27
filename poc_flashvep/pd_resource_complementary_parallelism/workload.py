"""Freeze genuine-source and controlled exact-token non-thinking traces."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from transformers import AutoTokenizer

MARKER = "PD_PAYLOAD_SENTINEL_DENSE_4210"
SEED = 20261128
LENGTHS = (2048, 4096, 8192, 16384, 32768)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def raw_question(text):
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
    return match[1] if match else text


def chat_boundary(tokenizer):
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": MARKER}], add_generation_prompt=True,
        tokenize=False, enable_thinking=False)
    head, end = template.split(MARKER)
    return (tokenizer(head, add_special_tokens=False).input_ids,
            tokenizer(end, add_special_tokens=False).input_ids, sha(template.encode()))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def trace_rows(source, ids, bundle, lengths, outputs, limit, density):
    head, end, _ = bundle
    seed = random.Random(SEED)
    time = 0.0
    for index in range(limit):
        item = source[index % len(source)]
        target = lengths[index % len(lengths)]
        payload = target - len(head) - len(end)
        body = ids[index % len(ids)]
        extensions = 0
        while len(body) < payload:
            another = ids[(index + extensions + 1) % len(ids)]
            body = body + another
            extensions += 1
        tokens = head + body[:payload] + end
        assert len(tokens) == target
        if index:
            time += seed.expovariate(density)
        yield {
            "request_id": f"{index:03d}/{item['request_id']}", "prompt_token_ids": tokens,
            "arrival_unit_s": time, "output_tokens": outputs[index % len(outputs)],
            "input_tokens": len(tokens), "corpus_extensions": extensions,
            "input_sha256": sha(json.dumps(tokens, separators=(",", ":")).encode()),
            "source_id": item["request_id"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[24, 64, 128])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_text())["requests"]
    random.Random(SEED + 1).shuffle(source)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = [raw_question(row["prompt"]) for row in source]
    ids = [tokenizer(prompt, add_special_tokens=False).input_ids for prompt in text]
    bundle = chat_boundary(tokenizer)
    receipts = {}
    shapes = [(f"{length}x{out}", (length,), (out,))
              for length in LENGTHS for out in (16, 32, 64, 128, 256)]
    shapes += [("mixed", (2048, 4096, 8192, 16384, 32768), (16, 32, 64, 128, 256)),
               ("mixed-heavy", (8192, 16384, 32768, 16384), (16, 32, 16, 64)),
               ("natural", (4096, 8192, 16384, 8192), (128,))]
    for name, lengths, outputs in shapes:
        for limit in args.requests:
            rows = list(trace_rows(source, ids, bundle, lengths, outputs, limit, 1.0))
            path = args.out / f"{name}-n{limit}.json"
            write(path, {"requests": rows, "seed": SEED, "template_sha256": bundle[-1],
                         "source_sha256": sha(args.source.read_bytes()), "regime": "controlled_longbench",
                         "unique_token_sequences": len({r["input_sha256"] for r in rows}),
                         "nonthinking": True})
            receipts[path.name] = {"hash": sha(path.read_bytes()), "count": len(rows),
                                   "extended_count": sum(r["corpus_extensions"] > 0 for r in rows)}
    write(args.out / "receipts.json", receipts)


if __name__ == "__main__":
    main()
