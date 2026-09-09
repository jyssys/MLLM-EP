"""CPU-only real four-image / natural-text volume-matched transfer controls.

No image upsampling, synthetic routes, or old performance reuse. This extends
workload coverage; it does not turn vLLM into any of the three native systems.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(2)
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--text-source", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = processor.tokenizer
    rows = [json.loads(s) for s in args.source.read_text().splitlines()]
    images = sorted({s for r in rows for s in r["images"] if "chartqa_" in s})
    assert len(images) >= 8
    texts = [json.loads(s)["text"] for s in args.text_source.read_text().splitlines()]
    assert texts
    max_pixels = 1024*1024

    def rendered(text, pictures):
        content = [{"type": "image", "image": image} for image in pictures]
        content.append({"type": "text", "text": text})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                              tokenize=False, add_generation_prompt=True)

    output, metadata = [], []
    for index in range(8):
        chosen = [images[(index+i)%len(images)] for i in range(4)]
        pictures = [Image.open(s).convert("RGB") for s in chosen]
        natural = "\n\n".join(texts[(index+i)%len(texts)] for i in range(32))
        ids = tokenizer.encode(natural, add_special_tokens=False)
        assert len(ids) >= 6000
        for long_context in (False, True):
            suffix = "Compare these four charts. Identify their main labels, trends, and any important differences."
            question = (("Background reading:\n" + tokenizer.decode(ids[:2048]) + "\n\n")
                        if long_context else "") + suffix
            encoded = processor(text=rendered(question, pictures), images=pictures,
                                images_kwargs={"max_pixels": max_pixels}, return_tensors="pt")
            total = encoded["input_ids"].shape[1]
            grid = encoded["image_grid_thw"].tolist()
            merge = processor.image_processor.merge_size
            vision = sum(t*h*w//(merge*merge) for t,h,w in grid)
            assert total+32 < 8192
            label = "long" if long_context else "short"
            text_prefix = "Read the following passage and briefly summarize it.\n"
            budget = total-len(tokenizer.encode(rendered(text_prefix, []), add_special_tokens=False))
            control = text_prefix+tokenizer.decode(ids[:budget])
            # Correct edge tokenization without inventing or changing routes.
            for _ in range(4):
                text_total = len(tokenizer.encode(rendered(control, []), add_special_tokens=False))
                if abs(text_total-total) <= 3:
                    break
                budget += total-text_total
                control = text_prefix+tokenizer.decode(ids[:budget])
            assert abs(text_total-total)/total <= .01
            for kind, question_text, paths in (("four_real_charts", question, chosen),
                                                ("matched_natural_text", control, [])):
                request_id = f"{kind}_{label}_{index:02d}"
                output.append({"request_id": request_id, "source_request_id": f"large_control_{index:02d}",
                               "family": f"{kind}_{label}", "question": question_text,
                               "images": paths, "max_new_tokens": 32, "ignore_eos": True})
                metadata.append({"request_id": request_id, "matched_pair": f"{label}_{index:02d}",
                                 "processor_prompt_tokens": total if paths else text_total,
                                 "vision_tokens": vision if paths else 0,
                                 "image_grid_thw": grid if paths else [], "images": paths,
                                 "image_sizes": [list(im.size) for im in pictures] if paths else [],
                                 "text_source_start_index": index,
                                 "text_source": str(args.text_source), "no_upsampling": True})
        for im in pictures:
            im.close()
    (args.out/"requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in output))
    (args.out/"manifest.json").write_text(json.dumps({"model": args.model, "max_pixels": max_pixels,
        "requests": metadata, "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(args.text_source.read_bytes()).hexdigest(),
        "scope": "new real-image inputs and matched natural text; not prior latency reuse"}, indent=2))
    assert not torch.cuda.is_initialized()
    print(json.dumps({"requests": len(output), "prompt_tokens": sorted({r["processor_prompt_tokens"] for r in metadata}),
                      "vision_tokens": sorted({r["vision_tokens"] for r in metadata}), "cuda_initialized": False}))


if __name__ == "__main__":
    main()
