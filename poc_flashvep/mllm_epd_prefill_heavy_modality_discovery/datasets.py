"""Freeze media-qualified natural cohorts and exact-length Qwen2.5-VL controls."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re


SEED = 2026112807
CUTOFF = 32768
LENGTHS = (4096, 8192, 16384, 24576, 32512)
FRACTIONS = (0, .25, .50, .75, .95)
IMAGE_TOKEN = "<|image_pad|>"
QUESTION = "\n\nSummarize the main point of the reference passage in one sentence."
NOTE = "\n\nReference passage:\n"


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=None)
def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return digest(payload)


def freeze(path, document):
    payload = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen workload would change at {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return digest(payload)


def ordered_hash(row):
    return digest(f"{SEED}:{row['request_id']}".encode())


def safe_rows(rows, check_hashes):
    selected, missing = [], []
    for row in rows:
        images = row["images"]
        tensor = Path(row["token_metadata_file"])
        files = [(Path(item["path"]), item["SHA256"]) for item in images]
        files.append((tensor, row["token_metadata_SHA256"]))
        if not images or not all(path.is_file() for path, _ in files):
            missing.append(row["request_id"])
            continue
        if check_hashes:
            for path, expected in files:
                if file_digest(path) != expected:
                    raise ValueError(f"Content mismatch for {row['request_id']} at {path}")
        if sum(image["visual_tokens"] for image in images) != row["visual_tokens"]:
            raise ValueError(f"Visual row count mismatch for {row['request_id']}")
        if row["visual_tokens"] + row["text_tokens"] + row["special_tokens"] != row["total_input_tokens"]:
            raise ValueError(f"LM token count mismatch for {row['request_id']}")
        if row["image_count"] != len(images):
            raise ValueError(f"Image cardinality mismatch for {row['request_id']}")
        selected.append(row)
    return selected, missing


def balanced(rows, count):
    """Select a nested roster approximately proportional to observed strata."""
    groups = defaultdict(list)
    for row in sorted(rows, key=ordered_hash):
        groups[row["stratum"]].append(row)
    positions = []
    for key, group in sorted(groups.items()):
        for index, row in enumerate(group):
            positions.append(((index + .5) / len(group), key, row))
    selected = [entry[2] for entry in sorted(positions, key=lambda item: item[:2])[:count]]
    return sorted(selected, key=ordered_hash)


def media_statistics(rows):
    def stats(values):
        values = sorted(values)
        if not values:
            return {}
        answer = {"min": values[0], "max": values[-1], "mean": sum(values) / len(values)}
        for p in (50, 95):
            i = (len(values) - 1) * p / 100
            lo = int(i)
            hi = min(lo + 1, len(values) - 1)
            answer[f"p{p}"] = values[lo] + (values[hi] - values[lo]) * (i - lo)
        return answer

    return {
        "requests": len(rows),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "images": stats([row["image_count"] for row in rows]),
        "vision_patches": stats([sum(a*b*c for a,b,c in row["grid_thw"]) for row in rows]),
        "visual_tokens": stats([row["visual_tokens"] for row in rows]),
        "text_tokens": stats([row["text_tokens"] for row in rows]),
        "structural_tokens": stats([row["special_tokens"] for row in rows]),
        "input_tokens": stats([row["total_input_tokens"] for row in rows]),
        "visual_ratio": stats([row["visual_tokens"] / row["total_input_tokens"] for row in rows]),
    }


def export(name, rows, mode, out, receipt, extra=None, preserve_order=False):
    if not rows:
        return
    rows = list(rows) if preserve_order else sorted(rows, key=ordered_hash)
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate request in {name}")
    summary = media_statistics(rows)
    extra = extra or {}
    file = out / f"{name}.json"
    sha256 = freeze(file, {"name": name, "seed": SEED, "workload_kind": mode,
                          "actual_count": len(rows), **extra, "requests": rows})
    receipt[name] = {"filename": file.name, "sha256": sha256, "workload_kind": mode, **extra, **summary,
                     "request_ids": [row["request_id"] for row in rows]}


def match_natural_image_count(rows, out, receipt):
    """Pair natural requests by LM input length while varying image geometry."""
    for label, low, high, few_limit in (("mid", 14000, 21000, 4), ("long", 18000, CUTOFF, 8)):
        eligible = [row for row in rows if low <= row["total_input_tokens"] <= high]
        few = [row for row in eligible if row["image_count"] <= few_limit]
        many = [row for row in eligible if row["image_count"] >= 17]
        options = sorted((abs(a["total_input_tokens"] - b["total_input_tokens"]),
                          a["request_id"], b["request_id"], a, b) for a in few for b in many)
        selected, used = [], set()
        for gap, left, right, a, b in options:
            if left in used or right in used:
                continue
            selected.append((gap, a, b))
            used.update((left, right))
            if len(selected) == 24:
                break
        pair_deltas = [{"few": a["request_id"], "many": b["request_id"], "input_gap_rows": gap}
                       for gap, a, b in selected]
        for name, index in (("few", 1), ("many", 2)):
            export(f"main400_matched_{label}_{name}_n{len(selected)}",
                   [pair[index] for pair in selected], "natural", out, receipt,
                   {"matched_input_range": [low, high], "pairs": pair_deltas}, preserve_order=True)


def build_natural(main100, natural149, complete, out, receipt):
    export("main100", main100, "natural", out, receipt, preserve_order=True)
    export("natural149", natural149, "natural", out, receipt, preserve_order=True)
    if len(complete) > len(natural149):
        export(f"natural_media_n{len(complete)}", complete, "natural", out, receipt, preserve_order=True)
    if len(complete) == 400:
        for count in (24, 64, 128):
            export(f"main400_mix_n{count}", balanced(complete, count), "natural", out, receipt)
        high = sorted(complete, key=lambda row: (-row["total_input_tokens"], ordered_hash(row)))[:128]
        for count in (24, 64, 128):
            export(f"main400_upper_n{count}", balanced(high, count), "natural", out, receipt,
                   {"minimum_upper_pool_input_tokens": min(row["total_input_tokens"] for row in high)})
        match_natural_image_count(complete, out, receipt)
    # These unchanged IDs form a fixed anchor independent of incremental media recovery.
    for count in (24, 64, 128):
        export(f"natural_mix_n{count}", balanced(natural149, count), "natural", out, receipt)
    longest = sorted(natural149, key=lambda row: (-row["total_input_tokens"], ordered_hash(row)))
    upper = longest[:64]
    for count in (24, 64):
        export(f"natural_upper_n{count}", balanced(upper, count), "natural", out, receipt,
               {"minimum_input_tokens": min(row["total_input_tokens"] for row in upper)})
    export("natural_very_long_n24", longest[:24], "natural", out, receipt)
    for name, lower, upper in (("one", 1, 1), ("two_four", 2, 4), ("five_eight", 5, 8),
                               ("nine_sixteen", 9, 16), ("seventeen_32", 17, 32), ("over32", 33, 200)):
        cohort = [row for row in natural149 if lower <= row["image_count"] <= upper]
        count = min(24, len(cohort))
        export(f"natural_{name}_n{count}", balanced(cohort, count), "natural", out, receipt,
               {"image_count_bounds":[lower, upper]})


def image_prefixes(rows):
    """Use contiguous, whole-image prefixes in original visual order."""
    seen = set()
    prefixes = []
    for row in sorted(rows, key=ordered_hash):
        visual = 0
        prefix = []
        for index, image in enumerate(row["images"], 1):
            prefix.append(image["SHA256"])
            visual += image["visual_tokens"]
            signature = tuple(prefix)
            if signature not in seen:
                seen.add(signature)
                prefixes.append((visual, row, index, digest("|".join(signature).encode())))
    return prefixes


def control_images(prefixes, target, fraction, count, limits=None):
    if fraction == 0:
        return [(0, None, 0, "")] * count
    candidates = []
    for visual, row, images, signature in prefixes:
        if limits is not None and not limits[0] <= images <= limits[1]:
            continue
        if abs(visual / target - fraction) <= .05 and visual + 2 * images + 80 < target:
            candidates.append((visual, row, images, signature))
    candidates.sort(key=lambda entry: (abs(entry[0]/target-fraction), entry[3]))
    # Prefer coverage across source requests, while keeping distinct visual sequences.
    ordered, spare, request_ids = [], [], set()
    for candidate in candidates:
        request_id = candidate[1]["request_id"]
        if request_id in request_ids:
            spare.append(candidate)
        else:
            ordered.append(candidate)
            request_ids.add(request_id)
    return (ordered + spare)[:count]


def large_image_triples(rows, count):
    """Construct a separately labeled image-count control at 8K visual rows."""
    gallery = {}
    for row in sorted(rows, key=ordered_hash):
        for image in row["images"]:
            if 1200 <= image["visual_tokens"] <= 6200:
                gallery.setdefault(image["SHA256"], (row, image))
    pairs = []
    for members in combinations(gallery.values(), 3):
        parents, images = zip(*members, strict=True)
        sources = [row["request_id"] for row in parents]
        if len(set(sources)) != 3:
            continue
        visual = sum(image["visual_tokens"] for image in images)
        if abs(visual - 8192) > 100:
            continue
        keys = tuple(image["SHA256"] for image in images)
        signature = digest("|".join(keys).encode())
        synthetic = {"images": images, "request_id": signature,
                     "source": "mixed/whole-image-control", "source_revision": None,
                     "image_source_requests": sources}
        pairs.append(((abs(visual - 8192), signature), keys, (visual, synthetic, 3, signature)))
    chosen, spare, seen = [], [], set()
    for _, keys, control in sorted(pairs):
        if any(key in seen for key in keys):
            spare.append(control)
        else:
            chosen.append(control)
            seen.update(keys)
    return (chosen + spare)[:count]


def base_passage(row):
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt"], re.DOTALL)
    return match.group(1) if match else row["prompt"]


def prompt_text(processor, images, question):
    structure = [{"type": "image"} for _ in images]
    structure.append({"type": "text", "text": question})
    return processor.apply_chat_template([{"role": "user", "content": structure}],
                                         tokenize=False, add_generation_prompt=True)


def expand_images(processor, prompt, grids):
    pieces = prompt.split(IMAGE_TOKEN)
    if len(pieces) != len(grids) + 1:
        raise ValueError("Prompt and image-grid cardinalities differ")
    return "".join(part + processor.replace_image_token({"image_grid_thw": grids}, j)
                   for j, part in enumerate(pieces[:-1])) + pieces[-1]


def prepare_control(processor, tokenizer, control, passage_tokens, target, initial):
    import numpy as np

    visual, owner, image_count, _ = control
    images = owner["images"][:image_count] if owner else []
    grids = np.asarray([image["grid_thw"] for image in images], dtype="int64").reshape((-1, 3))
    available = len(passage_tokens)
    candidate = None
    clip = min(available, max(0, target - visual - 2 * image_count - 44))
    tried = set()
    for _ in range(10):
        if clip in tried:
            break
        tried.add(clip)
        passage = tokenizer.decode(passage_tokens[:clip], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        question = NOTE + passage + QUESTION
        prompt = prompt_text(processor, images, question)
        full = expand_images(processor, prompt, grids)
        ids = tokenizer(full, add_special_tokens=False, verbose=False)["input_ids"]
        request = (question, prompt, ids)
        if candidate is None or abs(len(ids) - target) < abs(len(candidate[2]) - target):
            candidate = request
        if len(ids) == target:
            break
        clip = min(available, max(0, clip + target - len(ids)))
    if candidate is None or abs(len(candidate[2]) - target) > 2:
        raise ValueError(f"Failed to match input length in controlled {initial}: {target}")
    question, prompt, ids = candidate
    starts = [i for i, ident in enumerate(ids) if ident == processor.image_token_id and
              (i == 0 or ids[i-1] != ident)]
    ends = [i+1 for i, ident in enumerate(ids) if ident == processor.image_token_id and
            (i+1 == len(ids) or ids[i+1] != ident)]
    spans = list(map(list, zip(starts, ends, strict=True)))
    if [last-first for first,last in spans] != [i["visual_tokens"] for i in images]:
        raise ValueError(f"Image slot geometry differs from the previously verified processor grids at {initial}")
    specials = set(tokenizer.all_special_ids) - {processor.image_token_id}
    text_special = sum(t in specials for t in ids)
    revision = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    return {"request_id":initial, "question":question, "prompt":prompt,
            "prompt_SHA256":digest(prompt.encode()), "full_input_ids_SHA256":digest(np.asarray(ids,dtype="<i8").tobytes()),
            "images":[{**im,"visual_span":span} for im,span in zip(images,spans,strict=True)],
            "grid_thw":grids.tolist(), "image_visual_spans":spans, "image_count":image_count,
            "visual_tokens":visual, "text_tokens":len(ids)-visual-text_special, "special_tokens":text_special,
            "total_input_tokens":len(ids), "source":owner["source"] if owner else "LongBench/text",
            "image_source_request":owner["request_id"] if owner else None,
            "source_revision":owner["source_revision"] if owner else None,
            "model":"Qwen/Qwen2.5-VL-7B-Instruct", "model_revision":revision, "processor_revision":revision,
            "reference_answers":[], "stratum":"controlled", "context_limit":CUTOFF,
            "output_reserve":256, "natural_output_tokens":None, "natural_output_status":"NOT_MEASURED",
            "evidence":"PROCESSOR_GEOMETRY_AND_TOKENIZER_MEASUREMENT"}


def mixed_natural(processor, passages, multimodal, out, receipt):
    """Combine whole original natural inputs from the two observed source rosters."""
    tokenizer = processor.tokenizer
    text_rows = []
    chosen = sorted(passages, key=ordered_hash)
    revision = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    for source in chosen[:24]:
        prompt = prompt_text(processor, [], base_passage(source))
        ids = tokenizer(prompt, add_special_tokens=False, verbose=False)["input_ids"]
        specials = sum(ident in tokenizer.all_special_ids for ident in ids)
        text_rows.append({
            "request_id": source["request_id"], "source":f"LongBench/{source['source']}",
            "source_revision":"5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
            "question":base_passage(source), "prompt":prompt, "prompt_SHA256":digest(prompt.encode()),
            "images":[], "image_count":0, "visual_tokens":0, "grid_thw":[], "image_visual_spans":[],
            "text_tokens":len(ids)-specials, "total_input_tokens":len(ids), "special_tokens":specials,
            "model":"Qwen/Qwen2.5-VL-7B-Instruct", "model_revision":revision, "processor_revision":revision,
            "reference_answers":source["reference_answers"], "stratum":"text_only",
            "context_limit":CUTOFF, "natural_output_tokens":None, "natural_output_status":"NOT_MEASURED",
            "evidence":"TOKENIZER_MEASUREMENT",
        })
    export("natural_mixed_traffic_n48", balanced(multimodal, 24)+text_rows, "mixed_natural", out, receipt,
           {"routed_text_only_requests":24, "routed_image_requests":24})


def build_controls(rows, model_dir, passage_source, out, receipt, cell_size, targets):
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer = processor.tokenizer
    passages = read(passage_source)["requests"]
    random.Random(SEED).shuffle(passages)
    encoded = [tokenizer(base_passage(row), add_special_tokens=False, verbose=False)["input_ids"] for row in passages]

    @lru_cache(maxsize=cell_size)
    def source_tokens(index, target):
        indexes = [index]
        tokens = list(encoded[index])
        while len(tokens) < target + 128:
            indexes.append((index + len(indexes)) % len(passages))
            tokens.extend(tokenizer("\n\n", add_special_tokens=False)["input_ids"])
            tokens.extend(encoded[indexes[-1]])
        return tokens, [passages[i]["request_id"] for i in indexes]

    prefixes = image_prefixes(rows)
    shapes = [(target, fraction, None) for target in targets for fraction in FRACTIONS]
    if 16384 in targets:
        shapes += [(16384, .50, (1, 4)), (16384, .50, (8, 200))]
    for target, fraction, limits in shapes:
        images = control_images(prefixes, target, fraction, cell_size, limits)
        suffix = f"_i{limits[0]}_{limits[1]}" if limits else ""
        label = f"controlled_{target}_v{int(fraction*100):02d}{suffix}"
        output = []
        for index, group in enumerate(images):
            passage_ids, text_sources = source_tokens(index, target)
            row = prepare_control(processor, tokenizer, group, passage_ids, target, f"{label}__{index:03d}")
            row["possible_text_source_requests"] = text_sources
            output.append(row)
        export(f"{label}_n{len(output)}", output, "controlled", out, receipt,
               {"target_input_tokens":target, "target_visual_ratio":fraction, "image_count_bounds":limits})
        print(label, len(output), flush=True)
    if 16384 in targets:
        label = "controlled_16384_v50_triplet"
        output = []
        for index, group in enumerate(large_image_triples(rows, cell_size)):
            passage_ids, text_sources = source_tokens(index, 16384)
            row = prepare_control(processor, tokenizer, group, passage_ids, 16384, f"{label}__{index:03d}")
            row["possible_text_source_requests"] = text_sources
            row["image_source_requests"] = group[1]["image_source_requests"]
            output.append(row)
        export(f"{label}_n{len(output)}", output, "controlled", out, receipt,
               {"target_input_tokens":16384, "target_visual_ratio":.50, "image_count_bounds":[3, 3]})
        print(label, len(output), flush=True)
        if cell_size == 24:
            mixed_natural(processor, passages, rows, out, receipt)


def audit_rows(rows):
    return [{"request_id":row["request_id"], "source":row["source"], "source_revision":row["source_revision"],
             "stratum":row["stratum"], "image_count":row["image_count"], "visual_tokens":row["visual_tokens"],
             "text_tokens":row["text_tokens"], "structural_tokens":row["special_tokens"],
             "input_tokens":row["total_input_tokens"], "visual_ratio":row["visual_tokens"]/row["total_input_tokens"],
             "vision_patches":sum(a*b*c for a,b,c in row["grid_thw"]),
             "images":[{"hash":i["SHA256"], "original_size":i["original_resolution"],
                         "processed_size":i["processed_resolution"], "grid":i["grid_thw"],
                         "visual_span":i["visual_span"]} for i in row["images"]]}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main100", type=Path, required=True)
    parser.add_argument("--natural149", type=Path, required=True)
    parser.add_argument("--recovered", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--passages", type=Path)
    parser.add_argument("--cell-size", type=int, default=24)
    parser.add_argument("--targets", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    if args.model_dir is not None and args.passages is None:
        parser.error("--model-dir requires --passages")
    if args.controls_only and args.model_dir is None:
        parser.error("--controls-only requires --model-dir and --passages")
    if not 1 <= args.cell_size <= 128 or any(not 256 <= n <= CUTOFF - 256 for n in args.targets):
        parser.error("Choose valid cohort sizes and reserve at least 256 context rows for generation")
    original = {label: read(path)["requests"] for label, path in
                (("main100", args.main100), ("natural149", args.natural149))}
    qualified = {}
    for name, rows in original.items():
        selected, missing = safe_rows(rows, args.verify_hashes)
        if missing:
            raise ValueError(f"Missing frozen media in {name}: {missing}")
        qualified[name] = selected
    recovered = qualified["natural149"] if not args.recovered else read(args.recovered)["requests"]
    complete, missing = safe_rows(recovered, args.verify_hashes)
    if missing:
        raise ValueError(f"Missing media in recovered cohort: {missing}")
    receipt_path = args.results / "workload_receipts.json"
    receipt = read(receipt_path) if receipt_path.is_file() else {}
    if not args.controls_only:
        build_natural(qualified["main100"], qualified["natural149"], complete, args.out, receipt)
        audit_path = args.results / "workload_modality_rows.json"
        write(audit_path, audit_rows(complete))
        print(f"Natural media-qualified requests: {len(complete)}", flush=True)
    if args.model_dir is not None:
        build_controls(qualified["natural149"], args.model_dir, args.passages,
                       args.out, receipt, args.cell_size, args.targets)
    write(receipt_path, receipt)


if __name__ == "__main__":
    main()
