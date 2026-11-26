"""Single-H200 Track B model probes using unchanged processor input and BF16 weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .visual_flow import FlowProbe, GENERATED, QUESTION, SPECIAL, TEXT, TOKEN_CLASSES, VISUAL, experiment_arms


MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, contents):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prompt_groups(row, processor, cpu):
    ids = cpu["input_ids"][0].tolist()
    spans = row["image_visual_spans"]
    visual = {token for begin, end in spans for token in range(begin, end)}
    expected_visual = {i for i, token in enumerate(ids) if token == processor.image_token_id}
    if visual != expected_visual or len(visual) != row["visual_tokens"]:
        raise ValueError("Frozen image spans disagree with the actual processor IDs")
    prompt = row["prompt"]
    pieces = prompt.split(processor.image_token)
    if len(pieces) != len(spans) + 1:
        raise ValueError("Frozen visual placeholder count changed")
    expanded = pieces[0] + "".join(
        processor.image_token * (span[1] - span[0]) + suffix
        for span, suffix in zip(spans, pieces[1:], strict=True)
    )
    begin = expanded.rfind(row["question"])
    if begin != -1:
        end = begin + len(row["question"])
        rule = "exact_rendered_question"
    else:
        tag = "<|im_start|>user\n"
        role = expanded.rfind(tag)
        begin = role + len(tag)
        end = expanded.find("<|im_end|>", begin)
        if role < 0 or end < 0:
            raise ValueError("Could not identify the frozen user turn")
        rule = "rendered_user_turn"
    specials = set(processor.tokenizer.all_special_ids)
    categories = []
    for index, (token, (left, right)) in enumerate(zip(ids, cpu["offset_mapping"][0].tolist(), strict=True)):
        if index in visual:
            categories.append(VISUAL)
        elif token in specials:
            categories.append(SPECIAL)
        elif right > begin and left < end and right > left:
            categories.append(QUESTION)
        else:
            categories.append(TEXT)
    if len(categories) != row["total_input_tokens"] or QUESTION not in categories:
        raise ValueError("Question/visual taxonomy does not cover original input")
    return categories, rule


class Runtime:
    def __init__(self, model_path, args):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

        self.device = torch.device("cuda")
        self.args = args
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Assign exactly one explicitly leased CUDA device")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        ).eval().to(self.device)
        self.text = self.model.model.language_model
        self.lm_head = self.model.lm_head
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.probe = FlowProbe(self.model, modeling_qwen2_5_vl, self.device, args.output_cap)
        eoses = self.model.generation_config.eos_token_id
        self.eos = {eoses} if isinstance(eoses, int) else set(eoses)
        if not self.eos:
            raise ValueError("Frozen model lacks a greedy EOS contract")

    def prepare(self, row, verify=False):
        from PIL import Image

        images = []
        for item in row["images"]:
            if digest(Path(item["path"])) != item["SHA256"]:
                raise ValueError("Frozen image digest changed")
            with Image.open(item["path"]) as original:
                images.append(original.convert("RGB"))
        cpu = self.processor(text=[row["prompt"]], images=images, return_tensors="pt", return_offsets_mapping=True)
        for image in images:
            image.close()
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_SHA256"]:
            raise ValueError("Frozen prompt digest changed")
        path = Path(row["token_metadata_file"])
        if digest(path) != row["token_metadata_SHA256"]:
            raise ValueError("Frozen token metadata digest changed")
        categories, rule = prompt_groups(row, self.processor, cpu)
        with np.load(path) as metadata:
            if not np.array_equal(cpu["input_ids"][0].numpy(), metadata["input_ids"].reshape(-1)):
                raise ValueError("Actual HF token IDs disagree with the frozen original")
            expected_positions = metadata["positions"].reshape(3, 1, -1)
            expected_delta = int(metadata["rope_delta"].reshape(-1)[0])
        if "mm_token_type_ids" not in cpu:
            raise ValueError("Pinned Qwen processor did not return modality metadata")
        args = {key: cpu[key].to(self.device) for key in ("input_ids", "image_grid_thw", "mm_token_type_ids")}
        pixels = cpu["pixel_values"].to(self.device, dtype=torch.bfloat16)
        if args["image_grid_thw"].tolist() != row["grid_thw"]:
            raise ValueError("Actual image grid differs from the frozen grid")
        fused = self.text.embed_tokens(args["input_ids"])
        visual = self.model.model.get_image_features(pixels, args["image_grid_thw"]).pooler_output
        visual = torch.cat(visual, dim=0).to(fused)
        image_mask, _ = self.model.model.get_placeholder_mask(args["input_ids"], inputs_embeds=fused, image_features=visual)
        fused = fused.masked_scatter(image_mask, visual)
        positions = self.model.model.compute_3d_position_ids(
            **args, video_grid_thw=None, inputs_embeds=fused, attention_mask=None, past_key_values=None,
        )
        rope_delta = int(self.model.model.rope_deltas.item())
        if rope_delta != expected_delta or not np.array_equal(positions.cpu().numpy(), expected_positions):
            raise ValueError("Actual HF MRoPE differs from frozen original")
        self.fused = fused
        self.positions = positions
        self.rope_delta = rope_delta
        self.prompt_len = fused.shape[1]
        self.probe.configure()
        self.probe.prepare(categories, row["image_visual_spans"])
        verification = None
        if verify:
            untouched = self.model(**args, pixel_values=pixels, output_hidden_states=False, use_cache=False, logits_to_keep=1)
            actual = self.teacher([0], None)[0]
            expected = untouched.logits[0, -1].float()
            verification = {
                "last_prompt_logits_exact": bool(torch.equal(actual, expected)),
                "same_last_prompt_argmax": int(actual.argmax()) == int(expected.argmax()),
                "last_prompt_logits_abs_max": float((actual - expected).abs().max()),
            }
            if not verification["last_prompt_logits_exact"]:
                raise ValueError(f"Isolated text+vision route changed original prompt logits: {verification}")
        return {
            "request_id": row["request_id"], "source": row.get("study_source", row["source"]),
            "dataset": row["source"], "split": row.get("study_split"), "prompt_tokens": self.prompt_len,
            "image_spans": row["image_visual_spans"], "image_grid_thw": row["grid_thw"],
            "query_heads": self.text.layers[0].self_attn.num_heads,
            "kv_heads": self.text.layers[0].self_attn.num_key_value_heads,
            "head_dim": self.text.layers[0].self_attn.head_dim,
            "counts": {name: categories.count(i) for i, name in enumerate(TOKEN_CLASSES)},
            "question_span_rule": rule, "rope_delta": rope_delta, "original_baseline_preflight": verification,
            "matched_pairs": list(zip(*self.probe.pairs_cpu, strict=True)),
        }

    def next_positions(self, offset, count):
        return (torch.arange(offset, offset + count, device=self.device) + self.rope_delta).view(1, 1, -1).expand(3, 1, -1)

    def teacher(self, reference_tokens, arm, observe=False):
        if not reference_tokens:
            raise ValueError("The baseline greedy sequence is required for teacher forcing")
        self.probe.configure(arm, len(reference_tokens) if observe else 0)
        self.probe.offset = 0
        new_count = len(reference_tokens) - 1
        if new_count:
            tokens = torch.tensor(reference_tokens[:-1], device=self.device, dtype=torch.long)[None, :]
            combined = torch.cat((self.fused, self.text.embed_tokens(tokens)), dim=1)
            positions = torch.cat((self.positions, self.next_positions(self.prompt_len, new_count)), dim=-1)
        else:
            combined, positions = self.fused, self.positions
        result = self.text(inputs_embeds=combined, position_ids=positions, use_cache=False)
        return self.lm_head(result.last_hidden_state[0, self.prompt_len - 1 :]).float()

    def natural(self, arm):
        self.probe.configure(arm)
        self.probe.offset = 0
        result = self.text(inputs_embeds=self.fused, position_ids=self.positions, use_cache=True)
        state = result.past_key_values
        hidden = result.last_hidden_state[0, -1].clone()
        del result
        out = []
        for step in range(self.args.output_cap):
            encoded = self.lm_head(hidden).argmax().reshape(1, 1)
            greedy = int(encoded.item())
            out.append(greedy)
            if greedy in self.eos or step + 1 == self.args.output_cap:
                break
            position = self.prompt_len + step
            self.probe.offset = position
            result = self.text(input_ids=encoded, position_ids=self.next_positions(position, 1),
                               past_key_values=state, use_cache=True)
            hidden = result.last_hidden_state[0, -1]
            state = result.past_key_values
        text = self.processor.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"token_ids": out, "text": text, "output_tokens": len(out),
                "finish_reason": "eos" if out[-1] in self.eos else "output_cap"}

    def run_row(self, row, teacher_reference=None, verify=False):
        metadata = self.prepare(row, verify=verify)
        baseline = self.natural(None)
        reference = (baseline["token_ids"] if teacher_reference is None else teacher_reference)[: self.args.teacher_steps]
        baseline_logits = self.teacher(reference, None)
        observed_logits = self.teacher(reference, None, observe=True)
        if not torch.equal(baseline_logits, observed_logits):
            raise RuntimeError("Read-only attention/hidden probes changed original teacher logits")
        flow = self.probe.attention_rows()
        baseline_logp = F.log_softmax(baseline_logits, dim=-1)
        indices = torch.tensor(reference, device=self.device, dtype=torch.long)[:, None]
        base_nll = -baseline_logp.gather(-1, indices).flatten()
        arms = []
        for arm in experiment_arms():
            if self.args.smoke and arm.start not in (0, 20):
                continue
            logits = self.teacher(reference, arm)
            logp = F.log_softmax(logits, dim=-1)
            nll_delta = -logp.gather(-1, indices).flatten() - base_nll
            kl = (baseline_logp.exp() * (baseline_logp - logp)).sum(dim=-1).clamp_min(0)
            removed_pairs = sum(int(x) for x in self.probe.kv_pairs_removed)
            no_mask_error = self.probe.kv_no_mask_relative_L2
            eligible = self.probe.dose_cpu if arm.category == "visual_dose55" else self.probe.pairs_cpu[
                0 if arm.category == "visual" else 1
            ]
            record = {
                "arm": arm.label, "operation": arm.operation, "category": arm.category,
                "evidence": "CAUSAL_INTERVENTION",
                "layers": [arm.start + 1, arm.stop], "controlled_prompt_rows": len(eligible),
                "teacher_reference_tokens": reference, "delta_nll_by_step": nll_delta.tolist(),
                "forward_kl_by_step": kl.tolist(), "mean_delta_nll": float(nll_delta.mean()),
                "mean_forward_kl": float(kl.mean()),
                "teacher_argmax_match": int((logp.argmax(-1) == baseline_logp.argmax(-1)).sum()),
                "teacher_masked_pairs_across_layers": removed_pairs,
                "kv_no_mask_relative_L2": float(no_mask_error) if no_mask_error is not None else None,
                "theoretical_visual_row_layer_fraction": (
                    len(eligible) * (arm.stop - arm.start) / (metadata["counts"]["visual"] * 28)
                    if arm.category.startswith("visual") and arm.operation != "kv" else None
                ),
            }
            if self.args.natural == "all" or (
                self.args.natural == "representative" and (arm.start in (0, 12, 24) or arm.category == "visual_dose55")
            ):
                natural = self.natural(arm)
                new_output = natural["token_ids"]
                baseline_output = baseline["token_ids"]
                first = next((i for i, (a, b) in enumerate(zip(new_output, baseline_output)) if a != b),
                             min(len(new_output), len(baseline_output)))
                record.update(natural=natural, natural_exact=new_output == baseline_output,
                              natural_first_difference=first if new_output != baseline_output else None)
            arms.append(record)
        return {
            **metadata, "evidence": "DIAGNOSTIC", "measured_backend": "H200_HF_BF16_SDPA", "baseline": baseline,
            "original_teacher_reference_tokens": reference, "baseline_teacher_nll": base_nll.tolist(),
            "teacher_reference_source": "HF_own_greedy" if teacher_reference is None else "external_frozen_greedy",
            "attention_read_only_exact": True, "sampled_query_counts": {
                target: sum(item["layer"] == 1 and item["query_type"] == target for item in flow) for target in TOKEN_CLASSES
            }, "b1": flow, "b2": arms,
        }


def rows_from(manifest, args):
    rows = json.loads(manifest.read_text())["requests"]
    if args.request:
        wanted = set(args.request)
        rows = [row for row in rows if row["request_id"] in wanted]
        if len(rows) != len(wanted):
            raise ValueError("Requested rows are outside the frozen manifest")
    return rows[args.shard_index :: args.shard_count]


def load_teacher_reference(path, frozen):
    targets = {}
    known = {row["request_id"]: row for row in frozen}
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            request_id = item["request_id"]
            if request_id not in known:
                continue
            row = known[request_id]
            if (request_id in targets or not item["token_ids"] or item.get("prompt_SHA256") != row["prompt_SHA256"]
                    or item.get("input_tokens_manifest") != row["total_input_tokens"]):
                raise ValueError(f"External teacher identity mismatch for {request_id}")
            targets[request_id] = item["token_ids"]
    if targets.keys() != known.keys():
        raise ValueError("External teacher ledger is missing requests from this frozen shard")
    return targets


def main():
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--teacher-steps", type=int, default=32)
    parser.add_argument("--output-cap", type=int, default=2048)
    parser.add_argument("--teacher-reference", type=Path, help="Optional original client JSONL with frozen token IDs")
    parser.add_argument("--natural", choices=("all", "representative", "none"), default="all")
    parser.add_argument("--request", action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.model.name != MODEL_REVISION or not 1 <= args.teacher_steps <= args.output_cap <= 2048:
        parser.error("Use the exact pinned model, teacher prefix and Natural EOS cap")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard")
    if args.output_cap != 2048 and not args.smoke:
        parser.error("A reduced output cap is confined to explicit smoke tests")
    rows = rows_from(args.manifest, args)
    if not rows:
        parser.error("Empty frozen cohort")
    teacher = {} if args.teacher_reference is None else load_teacher_reference(args.teacher_reference, rows)
    args.run.mkdir(parents=True, exist_ok=args.resume)
    configuration = {
        "model_revision": MODEL_REVISION, "manifest_SHA256": digest(args.manifest), "source_SHA256": {
            name: digest(Path(__file__).with_name(name)) for name in ("track_b.py", "visual_flow.py")
        }, "torch": torch.__version__, "dataset_ids": [row["request_id"] for row in rows],
        "device_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_name": torch.cuda.get_device_name(),
        "output_cap": args.output_cap, "teacher_steps": args.teacher_steps, "natural_arms": args.natural,
        "torch_dtype": "BF16", "attention": "sdpa", "smoke": args.smoke, "transformers": transformers.__version__,
        "torch_cuda": torch.version.cuda,
        "external_teacher_SHA256": digest(args.teacher_reference) if args.teacher_reference is not None else None,
    }
    config = args.run / "config.json"
    if config.exists():
        if not args.resume or json.loads(config.read_text()) != configuration:
            raise RuntimeError("Refusing to mix unlike experiments")
    else:
        write(config, configuration)
    raw = args.run / "requests"
    raw.mkdir(exist_ok=True)
    failed = []
    with torch.inference_mode():
        runtime = Runtime(args.model, args)
        first = True
        with runtime.probe.install():
            for index, row in enumerate(rows):
                output = raw / f"{index:03d}.json"
                try:
                    if output.exists():
                        continue
                    if row["total_input_tokens"] + args.output_cap > row.get("context_limit", 32768):
                        raise ValueError("Original prompt and probe cap exceed frozen context allowance")
                    begin = time.monotonic()
                    data = runtime.run_row(row, teacher.get(row["request_id"]), verify=first)
                    first = False
                    data["elapsed_seconds_diagnostic"] = time.monotonic() - begin
                    write(output, data)
                    print("TRACK_B_REQUEST", index, row["request_id"], len(data["b1"]), len(data["b2"]), flush=True)
                except Exception as error:
                    import traceback

                    traceback.print_exc()
                    failed.append({"request_id": row["request_id"], "error": f"{type(error).__name__}: {error}"})
                    write(args.run / "failures.json", failed)
                    if first:
                        raise
    completed = len(list(raw.glob("*.json")))
    write(args.run / "complete.json", {"completed": completed, "planned": len(rows), "failures": failed})
    if completed != len(rows):
        raise RuntimeError("Track B finished with failed requests")


if __name__ == "__main__":
    main()
