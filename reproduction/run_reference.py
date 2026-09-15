"""Faithful Hugging Face LLaDA2.0-mini generation, with observational counters.

Each worker owns one physical GPU in CUDA_VISIBLE_DEVICES=4,5,6,7. There is
no TP, EP, decoder patch, kernel replacement, or threshold sweep here.
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "inclusionAI/LLaDA2.0-mini"
REVISION = "dad945cac317da394b390f82c7b40691d8a881ed"
SNAPSHOT = Path(
    "/home/esjung/.cache/huggingface/hub/models--inclusionAI--LLaDA2.0-mini"
    f"/snapshots/{REVISION}"
)

SANITY_PROMPTS = [
    "What is 7 + 8? Give the answer.",
    "What is 42 - 17? Give the answer.",
    "If 3 notebooks cost $12, what do 5 notebooks cost? Explain briefly.",
    "A shop sells 4 apples for $6. How much do 10 apples cost?",
    "Mina has 12 stickers, gives away 5, then receives 9. How many now?",
    "Calculate 18 × 7 and state the final number.",
    "A train goes 60 miles in 2 hours. At the same speed, how far in 5 hours?",
    "If all cats are mammals and Luna is a cat, is Luna a mammal? Why?",
    "Explain in one sentence why a dropped ball falls.",
    "Write a polite two-sentence email declining a meeting invitation.",
]


def parse_gsm8k(text):
    patterns = (
        r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:final answer|the answer is|answer:)\s*[^\n\d-]*([-+]?\$?\d[\d,]*(?:\.\d+)?)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].replace("$", "").replace(",", ""), pattern
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", text)
    return (matches[-1].replace("$", "").replace(",", ""), "last-number") if matches else (None, "none")


def gsm8k_prompt(question, protocol, yaml_path):
    if protocol == "raw-question":
        return question
    if protocol != "dinfer-fourshot":
        raise ValueError(protocol)
    task = yaml.safe_load(yaml_path.read_text())
    examples = task["fewshot_config"]["samples"][:4]
    prefix = "\n\n".join(
        f"Question: {example['question']}\nLet's think step by step\nAnswer: {example['answer']}"
        for example in examples
    )
    return prefix + f"\n\nQuestion: {question}\nLet's think step by step\nAnswer:"


def normalize_generated_ids(generated, input_ids):
    """Strip only the official early-EOS return's prompt prefix after generation."""
    returned_prompt = (
        generated.shape[1] >= input_ids.shape[1]
        and torch.equal(generated[0, : input_ids.shape[1]], input_ids[0])
    )
    ids = generated[0, input_ids.shape[1] :].tolist() if returned_prompt else generated[0].tolist()
    return ids, returned_prompt


def samples(task, source, protocol, yaml_path):
    if task == "sanity":
        return [{"id": i, "question": prompt, "answer": None} for i, prompt in enumerate(SANITY_PROMPTS)]
    rows = json.loads(source.read_text())
    if task == "gsm8k":
        return [
            {"id": int(row["id"]), "question": row["question"],
             "answer": row["answer"],
             "raw_prompt": gsm8k_prompt(row["question"], protocol, yaml_path)}
            for row in rows
        ]
    if task == "humaneval":
        return [
            {"id": int(row["id"]), "question": row["prompt"],
             "answer": None, "raw_prompt": row["prompt"],
             "test": row["test"], "entry_point": row["entry_point"]}
            for row in rows
        ]
    raise ValueError(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("sanity", "gsm8k", "humaneval"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("raw-question", "dinfer-fourshot"), default="raw-question")
    parser.add_argument("--yaml", type=Path, default=Path("/home/esjung/external/dinfer-llada2-flash-poc/evaluations/tasks/gsm8k/gsm8k-llada-mini.yaml"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--physical-gpu", type=int, choices=(4, 5, 6, 7),
                        help="optional selected-GPU override for a single fixed diagnostic sample")
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("This PoC may expose physical GPUs 4,5,6,7 only")
    if not 0 <= args.worker < args.workers <= 4:
        raise ValueError("invalid independent evaluation worker assignment")
    if args.task != "sanity" and args.source is None:
        raise ValueError("--source is required for benchmarks")
    if args.threshold != 0.95 or args.steps != 32:
        raise ValueError("Phase 1 uses the official model-card threshold/steps anchor only")
    if args.gen_length % 32:
        raise ValueError("gen-length must be a multiple of block length 32")

    chosen = samples(args.task, args.source, args.protocol, args.yaml)
    chosen = chosen[args.start : args.start + args.limit]
    chosen = [row for row in chosen if row["id"] % args.workers == args.worker]
    physical_gpu = args.physical_gpu if args.physical_gpu is not None else 4 + args.worker
    device = f"cuda:{physical_gpu - 4}"
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, trust_remote_code=True, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    actual_attention = model.config._attn_implementation
    forward = model.forward
    counter = {"nfe": 0}

    def counted_forward(*forward_args, **forward_kwargs):
        counter["nfe"] += 1
        return forward(*forward_args, **forward_kwargs)

    model.forward = counted_forward  # observation only; no tensor or decoder changes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        previous = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
        if any(record["gen_length_cap"] != args.gen_length or record["protocol"] != args.protocol
               or record["task"] != args.task for record in previous):
            raise RuntimeError("refusing to mix decoder/prompt protocols in one artifact")
        completed = {record["sample_id"] for record in previous}
    print(json.dumps({"model": MODEL_ID, "revision": REVISION, "device": device,
                      "physical_gpu": physical_gpu, "dtype": str(next(model.parameters()).dtype),
                      "attention": actual_attention, "samples": len(chosen),
                      "mask_id": tokenizer.mask_token_id, "eos_id": tokenizer.eos_token_id}), flush=True)
    for row in chosen:
        if row["id"] in completed:
            continue
        raw_prompt = row.get("raw_prompt", row["question"])
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        torch.manual_seed(20260915 + row["id"])
        counter["nfe"] = 0
        with torch.inference_mode():
            generated = model.generate(
                inputs=input_ids, temperature=0.0, threshold=args.threshold,
                block_length=32, steps=args.steps, gen_length=args.gen_length,
                eos_early_stop=True,
            )
        # The official eos_early_stop return branch yields final_x INCLUDING
        # input_ids, whereas its ordinary return branch yields only new IDs.
        # Normalize that *returned tensor* for evaluation; no decoder or model
        # operation is altered. Keep the flag so the source behavior is auditable.
        ids, returned_prompt = normalize_generated_ids(generated, input_ids)
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        clean = tokenizer.decode(ids, skip_special_tokens=True)
        parsed, parser_rule = parse_gsm8k(clean)
        gold = row["answer"].split("####")[-1].strip().replace(",", "") if args.task == "gsm8k" else None
        eos = tokenizer.eos_token_id in ids
        masked = ids.count(tokenizer.mask_token_id)
        termination = "REMAINING_MASK" if masked else "EOS" if eos else "GEN_LENGTH_CAP"
        first_block = input_ids.shape[1] // 32
        last_block = (input_ids.shape[1] + max(1, len(ids)) - 1) // 32
        record = {
            "sample_id": row["id"], "task": args.task, "protocol": args.protocol,
            "raw_prompt": raw_prompt, "formatted_prompt": formatted,
            "raw_generation": raw, "postprocessed_generation": clean,
            "parsed_answer": parsed, "parser_rule": parser_rule,
            "ground_truth": gold, "correct": parsed == gold if gold is not None else None,
            "generation_tokens": len(ids), "num_blocks": last_block - first_block + 1,
            "nfe": counter["nfe"], "termination_reason": termination,
            "remaining_mask_tokens": masked, "prompt_tokens": input_ids.shape[1],
            "official_return_included_prompt": returned_prompt,
            "gen_length_cap": args.gen_length, "steps_per_block_max": args.steps,
            "worker_physical_gpu": physical_gpu,
        }
        if args.task == "humaneval":
            record.update(test=row["test"], entry_point=row["entry_point"], original_prompt=row["question"])
        with args.output.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{args.task} id={row['id']} tokens={len(ids)} nfe={counter['nfe']} "
              f"term={termination} parsed={parsed!r} gold={gold!r}", flush=True)
        # The pure-Python official MoE path allocates many variable expert-row
        # tensors. Return allocator cache between independent benchmark cases
        # to avoid cumulative HBM pressure; this cannot change model semantics.
        del generated, input_ids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
