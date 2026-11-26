"""Freeze Discovery200 without accessing serving or model-output results."""

import argparse
import ast
from collections import Counter, defaultdict, deque
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import string


SEED = 20261127
OLD_ORIGINAL_SHA = "910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1"
OLD_LOCAL_SHA = "1d7221b2391c5239ca0a85a109d3b93d46ab1a03c4a42f4caaf44d0a9a437151"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_INPUT = 32768 - 2048
QUOTAS = {
    "Main100": (100, 75, 12), "MMMU": (40, 30, 5),
    "MathVista": (30, 23, 4), "MMReason": (30, 22, 3),
}
REPOS = {"MMMU": "MMMU/MMMU", "MathVista": "AI4Math/MathVista", "MMReason": "HuanjinYao/MMReason"}
PINS = {
    "MMMU": ("876ce5cb130f7f7e290ce4d9984357737d4db5cf", "validation", "Apache-2.0"),
    "MathVista": ("2b6ad69445fbb5695c9b165475e8decdbeb97747", "testmini", "CC-BY-SA-4.0"),
    "MMReason": ("cc6077a553a47efa8a9d9a3c649827b7bc5c85b4", "testmini", "Apache-2.0"),
}
IMAGE_MARKER = re.compile(r"<image\s+(\d+)>", flags=re.IGNORECASE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    value_bytes = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == value_bytes, f"Frozen file would change: {path}")
    else:
        path.write_bytes(value_bytes)


def stable(kind, identity):
    return digest(f"{SEED}:{kind}:{identity}".encode())


def diversify(rows, kind, group):
    buckets = defaultdict(list)
    for row in rows:
        buckets[group(row)].append(row)
    ordered = [
        deque(sorted(buckets[key], key=lambda row: stable(kind, row["request_id"])))
        for key in sorted(buckets, key=lambda key: stable(kind, repr(key)))
    ]
    while ordered:
        for bucket in ordered:
            yield bucket.popleft()
        ordered = [bucket for bucket in ordered if bucket]


def geometry_group(row):
    image_bucket = 1 if row["image_count"] == 1 else 2 if row["image_count"] <= 4 else 3
    token_bucket = sum(row["total_input_tokens"] > value for value in (2048, 8192, 16384))
    return row["stratification_group"], image_bucket, token_bucket


def assignments(rows, study_source):
    total, n_discovery, n_tier = QUOTAS[study_source]
    require(len(rows) == total, f"{study_source} requires {total} rows")
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["stratification_group"]].append(row)
    # Quota remainder uses a stable hash after multiplying by the exact split fraction.
    allocation = {key: len(group) * n_discovery // total for key, group in by_key.items()}
    remainders = sorted(by_key, key=lambda key: (
        -(len(by_key[key]) * n_discovery % total), stable(f"{study_source}:split", repr(key))))
    for key in remainders[:n_discovery - sum(allocation.values())]:
        allocation[key] += 1
    discovery_ids = {
        row["request_id"] for key, members in by_key.items()
        for row in list(diversify(members, f"{study_source}:split:{key}", geometry_group))[:allocation[key]]
    }
    tier = list(diversify([row for row in rows if row["request_id"] in discovery_ids],
                          f"{study_source}:tier", geometry_group))[:n_tier]
    tier_ids = {row["request_id"] for row in tier}
    require(len(discovery_ids) == n_discovery and len(tier_ids) == n_tier, "Split count mismatch")
    return [{
        **row, "study_source": study_source,
        "study_split": "discovery" if row["request_id"] in discovery_ids else "validation",
        "tier1": row["request_id"] in tier_ids,
    } for row in rows]


def manifest(name, rows):
    return {"name": name, "seed": SEED, "actual_count": len(rows), "target_count": len(rows),
            "discovery_ids": [r["request_id"] for r in rows if r["study_split"] == "discovery"],
            "validation_ids": [r["request_id"] for r in rows if r["study_split"] == "validation"],
            "heldout_ids": [r["request_id"] for r in rows if r["study_split"] == "validation"],
            "tier1_ids": [r["request_id"] for r in rows if r["tier1"]],
            "requests": rows}


def old_rows(local):
    require(file_hash(local) == OLD_LOCAL_SHA, "Local relocated Main100 failed expected SHA256")
    rows = read(local)["requests"]
    for row in rows:
        require(all(Path(image["path"]).is_file() for image in row["images"]), f"Missing original media: {row['request_id']}")
        require(Path(row["token_metadata_file"]).is_file(), f"Missing frozen token metadata: {row['request_id']}")
    return assignments([{**row, "source_id": row["original_id"], "stratification_group": row["source"]}
                        for row in rows], "Main100")


def partial(output, rows):
    write(output / "main100_discovery.json", manifest("Main100 Discovery inputs", [r for r in rows if r["study_split"] == "discovery"]))
    write(output / "tier1_main100_12.json", manifest("Tier1 Main100 portion", [r for r in rows if r["tier1"]]))


def prepare_record(study_source, local, row_index, row):
    if study_source == "MMMU":
        identifier = row["id"]
        images = {i: row[f"image_{i}"] for i in range(1, 8) if row[f"image_{i}"] and row[f"image_{i}"]["bytes"]}
        group = local.parent.name
        choices = ast.literal_eval(row["options"] or "[]")
        if row["question_type"] == "multiple-choice":
            if not choices or row["answer"] not in string.ascii_uppercase[:len(choices)]:
                return None, "REFERENCE_OR_OPTIONS_INVALID"
            choice_text = "".join(f"({letter}) {text}\n" for letter, text in zip(string.ascii_uppercase, choices))
            prompt = f"{row['question']}\n\n{choice_text}\n\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = f"{row['question']}\n\nAnswer the question using a single word or phrase."
        markers = [int(item) for item in IMAGE_MARKER.findall(prompt)]
        if "<img=" in prompt or any(i not in images for i in markers) or set(markers) != set(images):
            return None, "IMAGE_REFERENCE_UNRESOLVED"
        pieces = []
        last = 0
        in_order = []
        for match in IMAGE_MARKER.finditer(prompt):
            if match.start() > last:
                pieces.append({"type": "text", "text": prompt[last:match.start()]})
            pieces.append({"type": "image", "image_index": len(in_order)})
            in_order.append(images[int(match.group(1))])
            last = match.end()
        if last < len(prompt):
            pieces.append({"type": "text", "text": prompt[last:]})
        metadata = {key: row[key] for key in ("question_type", "topic_difficulty", "subfield", "img_type")}
        contract = "MMMU official construct_prompt and llava1.5.yaml zero-shot wording with each image marker replaced in situ"
    elif study_source == "MathVista":
        identifier, in_order = row["pid"], [row["decoded_image"]]
        group = row["metadata"]["task"]
        choices, prompt = row["choices"] or [], row["query"]
        metadata = {**row["metadata"], **{key: row[key] for key in ("question_type", "answer_type", "unit", "precision")}}
        pieces = [{"type": "image", "image_index": 0}, {"type": "text", "text": prompt}]
        contract = "Published MathVista query with the provided hint and choices"
    else:
        identifier, in_order = row["idx"], [row["image"]]
        group = row["discipline"]
        choices, prompt = [], row["cot_question"]
        metadata = {"discipline": group, "question_type": "open-ended"}
        pieces = [{"type": "image", "image_index": 0}, {"type": "text", "text": prompt}]
        contract = "Author-provided MMReason cot_question from testmini"
    if not (str(row["answer"] or "").strip() and prompt and in_order):
        return None, "REFERENCE_OR_IMAGE_MISSING"
    if any(not image or not image.get("bytes") for image in in_order):
        return None, "IMAGE_BYTES_MISSING"
    revision, split, license_name = PINS[study_source]
    return ({
        "request_id": REPOS[study_source].replace("/", "__") + "__" + str(identifier),
        "source": REPOS[study_source], "source_id": str(identifier), "original_id": str(identifier),
        "source_revision": revision, "source_split": split, "source_license": license_name,
        "source_parquet": local.name, "source_row_index": row_index,
        "study_source": study_source, "stratification_group": group,
        "stratum": "single_image" if len(in_order) == 1 else "multi_image",
        "metadata": metadata, "question_original": row["question"], "question": prompt,
        "reference_answers": [row["answer"]], "options": choices, "content_order": pieces,
        "prompt_protocol": contract, "image_count": len(in_order), "page_count": None,
        "source_images": in_order,
    }, None)


def save_image(output, source_image):
    from PIL import Image

    blob = source_image["bytes"]
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        raw_hash = digest(blob)
        pixels = digest(str(image.size).encode() + image.convert("RGB").tobytes())
        extension = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff", "WEBP": ".webp", "GIF": ".gif"}.get(image.format, ".img")
        path = output / "media" / raw_hash[:2] / (raw_hash + extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(file_hash(path) == raw_hash, f"Existing image differs: {path}")
        else:
            path.write_bytes(blob)
        return {"path": str(path), "SHA256": raw_hash, "pixel_SHA256": pixels,
                "original_resolution": list(image.size), "original_name": source_image.get("path")}


def prepare(output):
    import pyarrow.parquet as pq

    files = read(output / "source/receipt.json")
    details = {}
    for study_source, repo in REPOS.items():
        rows, rejected = [], Counter()
        source_files = [record for record in files if record["repo_id"] == repo]
        for receipt in sorted(source_files, key=lambda item: item["filename"]):
            require(receipt["revision"] == PINS[study_source][0], "Dataset revision mismatch")
            local = output / "source" / repo / receipt["filename"]
            require(file_hash(local) == receipt["sha256"] and local.stat().st_size == receipt["bytes"],
                    f"Source integrity changed: {receipt['filename']}")
            row_index = 0
            for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
                for raw in batch.to_pylist():
                    row, reason = prepare_record(study_source, local, row_index, raw)
                    if row:
                        rows.append(row)
                    else:
                        rejected[reason] += 1
                    row_index += 1
        # Random tie order comes from the fixed hash seed. Use raw question
        # length and image count here; exact Qwen geometry is used in the split.
        candidates = list(diversify(rows, f"{study_source}:pool", lambda row: (
            row["stratification_group"], min(row["image_count"], 4),
            sum(len(row["question_original"]) > value for value in (160, 400, 800)),
        )))[:QUOTAS[study_source][0] + 20]
        for row in candidates:
            row["images"] = [save_image(output, image) for image in row.pop("source_images")]
        details[study_source] = {"eligible_source_rows": len(rows), "staged_rows": len(candidates),
                                 "excluded_before_staging": dict(rejected)}
        write(output / f"{study_source.lower()}_candidates.json", candidates)
        print(study_source, details[study_source], flush=True)
    write(output / "candidate_receipt.json", details)


def profile_rows(rows, output, model):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    import types

    torch.set_num_threads(1)
    proc = AutoProcessor.from_pretrained(model, local_files_only=True)
    rope = types.SimpleNamespace(config=AutoConfig.from_pretrained(model, local_files_only=True))
    rope.get_vision_position_ids = types.MethodType(Qwen2_5_VLModel.get_vision_position_ids, rope)
    tensor_folder = output / "token_metadata"
    tensor_folder.mkdir(exist_ok=True)
    special = set(proc.tokenizer.all_special_ids) - {proc.image_token_id}
    for row in rows:
        images = [Image.open(image["path"]).convert("RGB") for image in row["images"]]
        content = [dict(type="image") if item["type"] == "image" else item for item in row["content_order"]]
        prompt = proc.apply_chat_template([dict(role="user", content=content)], tokenize=False, add_generation_prompt=True)
        enc = proc(text=[prompt], images=images, return_tensors="pt", return_mm_token_type_ids=True)
        ids = enc["input_ids"][0]
        length = len(ids)
        if length > MAX_INPUT:
            yield None, "CONTEXT_OVERFLOW"
            continue
        grids = enc["image_grid_thw"].tolist()
        positions = np.flatnonzero((ids == proc.image_token_id).numpy())
        blocks = np.split(positions, np.where(np.diff(positions) != 1)[0] + 1)
        spans = [[int(block[0]), int(block[-1]) + 1] for block in blocks]
        require(len(spans) == len(grids) == len(images), "Image span or grid count did not match source images")
        visual = len(positions)
        n_special = sum(token in special for token in ids.tolist())
        for (left, right), grid, media in zip(spans, grids, row["images"], strict=True):
            require(right - left == int(np.prod(grid)) // 4, "Qwen grid and visual span disagree")
            media.update(grid_thw=grid, processed_resolution=[grid[2] * 14, grid[1] * 14],
                         visual_span=[left, right], visual_tokens=right - left)
        if "mm_token_type_ids" not in enc:
            enc["mm_token_type_ids"] = (enc["input_ids"] == proc.image_token_id).long()
        mrope, delta = Qwen2_5_VLModel.get_rope_index(
            rope, input_ids=enc["input_ids"], mm_token_type_ids=enc["mm_token_type_ids"],
            image_grid_thw=enc["image_grid_thw"], attention_mask=enc.get("attention_mask"))
        tensor_path = tensor_folder / (digest(row["request_id"].encode()) + ".npz")
        # Same NumPy format as the frozen Main100 metadata.
        tensor_data = io.BytesIO()
        np.savez_compressed(tensor_data, input_ids=ids.numpy(), image_grid_thw=np.asarray(grids),
                            positions=mrope.numpy(), rope_delta=delta.numpy(),
                            mm_token_type_ids=enc["mm_token_type_ids"].numpy())
        value = tensor_data.getvalue()
        if tensor_path.exists():
            require(file_hash(tensor_path) == digest(value), "Token metadata changed after freeze")
        else:
            tensor_path.write_bytes(value)
        yield ({
            **row, "prompt": prompt, "prompt_SHA256": digest(prompt.encode()),
            "prompt_template_version": "Discovery200 zero-shot per-source v1",
            "model": MODEL, "model_revision": MODEL_REVISION, "processor_revision": MODEL_REVISION,
            "total_input_tokens": length, "visual_tokens": visual, "special_tokens": n_special,
            "text_tokens": length - visual - n_special, "image_visual_spans": spans, "grid_thw": grids,
            "token_metadata_file": str(tensor_path), "token_metadata_SHA256": file_hash(tensor_path),
            "encoder_output_shape_structural": [visual, 3584], "encoder_output_shape_measured": None,
            "natural_output_tokens": None, "natural_output_status": "NOT_MEASURED",
            "context_limit": 32768, "output_reserve": 2048, "truncated": False,
            "eligible": True, "exclusion_reason": None, "evidence": "PROCESSOR_MEASUREMENT",
        }, None)


def content_key(row):
    question = " ".join((row.get("question_original") or row["question"]).split()).casefold()
    # Pixel contents allow equality across image encodings. Reused individual
    # images remain valid when the source question or ordered image tuple differs.
    image_hashes = tuple(item["pixel_SHA256"] for item in row["images"])
    return (question, image_hashes)


def freeze(output, local, model):
    require(model.name == MODEL_REVISION, "Use the pinned Qwen model snapshot")
    old = old_rows(local)
    registered = {row["request_id"] for row in old}
    contents = {content_key(row) for row in old}
    require(len(registered) == len(contents) == 100, "Frozen Main100 contains an exact duplicate")
    all_rows = list(old)
    excluded = []
    sources = {}
    for source, (count, _, _) in QUOTAS.items():
        if source == "Main100":
            continue
        selected = []
        candidates = read(output / f"{source.lower()}_candidates.json")
        for candidate, (row, reason) in zip(candidates, profile_rows(candidates, output, model)):
            if row is None:
                excluded.append({"request_id": candidate["request_id"], "reason": reason})
            elif row["request_id"] in registered or content_key(row) in contents:
                excluded.append({"request_id": row["request_id"], "reason": "EXACT_CONTENT_OR_ID_DUPLICATE"})
            else:
                registered.add(row["request_id"])
                contents.add(content_key(row))
                selected.append(row)
            if len(selected) == count:
                break
        require(len(selected) == count, f"{source} only supplied {len(selected)} eligible rows")
        assigned = assignments(selected, source)
        sources[source] = assigned
        all_rows.extend(assigned)
        print("FROZEN", source, len(assigned), min(r["total_input_tokens"] for r in assigned),
              max(r["total_input_tokens"] for r in assigned), flush=True)
    discovery = [row for row in all_rows if row["study_split"] == "discovery"]
    validation = [row for row in all_rows if row["study_split"] == "validation"]
    tier = [row for row in discovery if row["tier1"]]
    require((len(all_rows), len(discovery), len(validation), len(tier)) == (200, 150, 50, 24), "Final quota mismatch")
    require(len(registered) == len(contents) == 200 and not {r["request_id"] for r in discovery}
            & {r["request_id"] for r in validation}, "Final membership overlaps")
    files = {}
    for name, rows in (("discovery200", all_rows), ("discovery150", discovery), ("validation50", validation), ("tier1_24", tier)):
        name_path = output / (name + ".json")
        write(name_path, manifest(name, rows))
        files[name] = {"requests": len(rows), "SHA256": file_hash(name_path), "file": name_path.name}
    source_receipt = output / "source/receipt.json"
    write(output / "dataset_freeze_receipt.json", {
        "status": "FROZEN", "selection_seed": SEED, "selection_basis": "Source metadata and Qwen processor geometry only",
        "source_shard_receipt_SHA256": file_hash(source_receipt),
        "verified_source_shards": len(read(source_receipt)), "main100_original_SHA256": OLD_ORIGINAL_SHA,
        "main100_relocated_SHA256": OLD_LOCAL_SHA, "model_revision": MODEL_REVISION,
        "manifests": files, "excluded": excluded,
        "count_by_source_and_split": {
            source: {split: sum(r["study_split"] == split for r in rows) for split in ("discovery", "validation")}
            | {"tier1": sum(r["tier1"] for r in rows)}
            for source, rows in {"Main100": old, **sources}.items()
        },
    })



def isolation_keys(row):
    question, image_hashes = content_key(row)
    keys = {("context", digest(json.dumps(image_hashes).encode())), ("question", question),
            ("original_id", row["source"], row["source_id"])}
    for image in row["images"]:
        keys.update({("image", image["SHA256"]), ("pixel", image["pixel_SHA256"])})
    for key, value in (("document_id", row.get("document_id")),
                       ("document_sha", (row.get("document") or {}).get("document_SHA256"))):
        if value:
            keys.add((key, value))
    return keys


def connected_groups(rows):
    parents = list(range(len(rows)))
    owners = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        for key in sorted(isolation_keys(row)):
            earlier = owners.setdefault(key, index)
            parents[root(index)] = root(earlier)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    return list(groups.values())


def isolated_validation(groups, target, source):
    """Keep the original source count and minimize fold changes under group isolation."""
    ordered = sorted(groups, key=lambda group: stable("isolated:" + source, min(r["request_id"] for r in group)))
    plans = {0: (0, ())}
    for index, group in enumerate(ordered):
        if any(row["tier1"] for row in group):
            continue
        # Relative to sending the group to Discovery, cost is the net change
        # in the number of rows whose v1 fold label changes.
        change = sum(1 if row["study_split"] == "discovery" else -1 for row in group)
        for old_count, (old_cost, selected) in tuple(plans.items()):
            count = old_count + len(group)
            if count > target:
                continue
            proposal = old_cost + change, (*selected, index)
            if count not in plans or proposal < plans[count]:
                plans[count] = proposal
    require(target in plans, f"No group-isolated {source} split preserves its count and Tier1")
    return {row["request_id"] for index in plans[target][1] for row in ordered[index]}


def isolate(output, initial):
    manifest_v1 = initial / "discovery200.json"
    initial_receipt = read(initial / "dataset_freeze_receipt.json")
    require(initial_receipt["manifests"]["discovery200"]["SHA256"] == file_hash(manifest_v1),
            "Input v1 complete manifest hash changed")
    rows = read(manifest_v1)["requests"]
    groups = connected_groups(rows)
    per_source = defaultdict(list)
    for group in groups:
        sources = {row["source"] for row in group}
        require(len(sources) == 1, "A deduplicated group spans different source quotas")
        per_source[sources.pop()].append(group)
    validation = set()
    for source, components in sorted(per_source.items()):
        target = sum(row["study_split"] == "validation" for group in components for row in group)
        validation.update(isolated_validation(components, target, source))
    group_ids = {
        row["request_id"]: digest(json.dumps(sorted(member["request_id"] for member in group)).encode())
        for group in groups for row in group
    }
    moved = []
    for row in rows:
        assigned = "validation" if row["request_id"] in validation else "discovery"
        if row["study_split"] != assigned:
            moved.append({"request_id": row["request_id"], "v1": row["study_split"], "v2": assigned})
        row["study_split"], row["validation_group"] = assigned, group_ids[row["request_id"]]
    discovery_rows = [row for row in rows if row["study_split"] == "discovery"]
    validation_rows = [row for row in rows if row["study_split"] == "validation"]
    tier1 = [row for row in discovery_rows if row["tier1"]]
    overlap = set().union(*(isolation_keys(row) for row in discovery_rows)) & set().union(
        *(isolation_keys(row) for row in validation_rows))
    require(not overlap and (len(rows), len(discovery_rows), len(validation_rows), len(tier1)) == (200, 150, 50, 24),
            "Group isolation or exact quotas failed")
    require([row["request_id"] for row in tier1] == read(initial / "tier1_24.json")["tier1_ids"],
            "Tier1 changed during group isolation")

    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, selected in (("discovery200", rows), ("discovery150", discovery_rows),
                           ("validation50", validation_rows), ("tier1_24", tier1)):
        path = output / f"{name}.json"
        write(path, manifest(name, selected))
        files[name] = {"requests": len(selected), "SHA256": file_hash(path), "file": path.name}
    write(output / "dataset_freeze_receipt.json", {
        **initial_receipt,
        "selection_basis": "Same v1 rows with grouped folds by image bytes/pixels, ordered-image context, "
                           "document ID/hash, normalized original question, and source-qualified original ID.",
        "manifests": files, "v1_freeze_receipt_SHA256": file_hash(initial / "dataset_freeze_receipt.json"),
        "connected_groups": dict(sorted(Counter(len(group) for group in groups).items())),
        "cross_fold_keys": len(overlap), "tier1_same_ids_and_order": True, "fold_moves": moved,
    })


def export_registry(initial, output):
    source = initial / "source/receipt.json"
    v2 = initial / "grouped_v2"
    manifest_path = v2 / "discovery200.json"
    receipt = read(v2 / "dataset_freeze_receipt.json")
    require(file_hash(manifest_path) == receipt["manifests"]["discovery200"]["SHA256"], "Grouped manifest hash differs")
    require(file_hash(source) == receipt["source_shard_receipt_SHA256"], "Official-source receipt hash differs")
    rows = []
    for row in read(manifest_path)["requests"]:
        question, pixels = content_key(row)
        rows.append({
            "request_id": row["request_id"], "source_id": row["source_id"], "source": row["source"],
            "source_revision": row["source_revision"], "source_split": row.get("source_split"),
            "source_license": row.get("source_license"), "study_source": row["study_source"],
            "study_split": row["study_split"], "tier1": row["tier1"], "validation_group": row["validation_group"],
            "stratification_group": row["stratification_group"], "prompt_sha256": row["prompt_SHA256"],
            "original_question_sha256": digest(question.encode()),
            "context_sha256": digest(json.dumps(pixels).encode()),
            "image_sha256": " ".join(image["SHA256"] for image in row["images"]),
            "image_pixel_sha256": " ".join(pixels), "token_metadata_sha256": row["token_metadata_SHA256"],
            "input_tokens": row["total_input_tokens"], "visual_tokens": row["visual_tokens"],
            "text_tokens": row["text_tokens"], "image_count": row["image_count"],
        })
    output.mkdir(parents=True, exist_ok=True)
    write(output / "dataset_freeze_receipt.json", receipt)
    write(output / "official_source_shards.json", read(source))
    write(output / "preparation_receipt.json", read(initial / "candidate_receipt.json"))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    registry = output / "discovery200_registry.csv"
    value = buffer.getvalue().encode()
    if registry.exists():
        require(registry.read_bytes() == value, "Frozen publication registry would change")
    else:
        registry.write_bytes(value)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "freeze", "isolate", "export"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--main100", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "prepare":
        prepare(output)
    elif args.phase == "export":
        require(args.initial is not None, "Export needs the frozen data root")
        export_registry(args.initial.resolve(), output)
    elif args.phase == "isolate":
        require(args.initial and args.initial.resolve() != output, "Isolate needs a distinct frozen v1 source")
        isolate(output, args.initial.resolve())
    else:
        require(args.main100 and args.model, "Freeze needs the relocated Main100 and verified Qwen snapshot")
        freeze(output, args.main100.resolve(), args.model.resolve())


if __name__ == "__main__":
    main()
