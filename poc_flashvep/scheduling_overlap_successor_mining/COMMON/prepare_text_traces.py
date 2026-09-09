"""Deterministic real-content request traces, with explicit non-paper provenance.

Reuses archived FineWeb-Edu text, not archived measurements. Arrival schedules
are designed controls, not the paper's Azure trace. No CUDA initialization.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    assert not args.out.exists(), "Use a new trace version; do not overwrite arrivals"
    args.out.mkdir(parents=True)
    docs = [json.loads(s) for s in args.source.read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rng = random.Random(20260908)
    rng.shuffle(docs)
    encoded = [(d["id"], tok.encode(d["text"], add_special_tokens=False)) for d in docs]
    configurations = {
        "heterogeneous_steady": (48, [512, 2048, 6144, 8192], [64, 128, 256]),
        "heterogeneous_bursty": (48, [512, 2048, 6144, 8192], [64, 128, 256]),
        "short_high_concurrency": (512, [128, 256, 512], [128]),
        "warmup": (12, [256, 1024, 4096], [32]),
    }
    manifest = {"seed": 20260908, "text_source": str(args.source),
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "source_dataset": "HuggingFaceFW/fineweb-edu sample-10BT train archived rows",
                "arrival_source": "designed steady/bursty controls; NOT Azure paper reproduction",
                "tokenizer": args.tokenizer, "traces": {}}
    for name, (n, sizes, outputs) in configurations.items():
        rows = []
        for i in range(n):
            # The two heterogeneity controls have exactly the same request set.
            budget = sizes[i % len(sizes)]
            content_ids, provenance = [], []
            j = (i * 7) % len(encoded)
            while len(content_ids) < max(1, budget - 42):
                source_id, ids = encoded[j % len(encoded)]
                provenance.append(source_id)
                content_ids += ids + tok.encode("\n\n", add_special_tokens=False)
                j += 1
            context = tok.decode(content_ids[:max(1, budget - 42)])
            prompt = ("<|im_start|>user\nSummarize the following source text accurately. "
                      "Mention its main factual claims.\n\n" + context +
                      "<|im_end|>\n<|im_start|>assistant\n")
            arrival = (i * 0.5 if name == "heterogeneous_steady" else
                       (i // 16) * 10 + (i % 16) * 0.01
                       if name == "heterogeneous_bursty" else
                       i * 0.01 if name == "short_high_concurrency" else i * 0.2)
            rows.append({"request_id": f"{'heterogeneous' if name.startswith('hetero') else name}_{i:04d}",
                         "arrival_s": arrival, "prompt": prompt,
                         "prompt_tokens": len(tok.encode(prompt, add_special_tokens=False)),
                         "target_prompt_tokens": budget,
                         "max_new_tokens": outputs[i % len(outputs)],
                         "ignore_eos": True,
                         "protocol": "fixed-output scheduling control; correctness compared across configs",
                         "source_ids": provenance, "modality": "text"})
        path = args.out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["traces"][name] = {
            "requests": n, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "total_output_tokens": sum(r["max_new_tokens"] for r in rows),
            "last_arrival_s": rows[-1]["arrival_s"],
            "min_prompt_tokens": min(r["prompt_tokens"] for r in rows),
            "max_prompt_tokens": max(r["prompt_tokens"] for r in rows)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
